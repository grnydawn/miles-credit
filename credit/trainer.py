import gc
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
import optuna
from credit.models.checkpoint import TorchFSDPCheckpointIO


def cleanup():
    dist.destroy_process_group()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


class Trainer:

    def __init__(self, model, rank, module=False):
        super(Trainer, self).__init__()
        self.model = model
        self.rank = rank
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

        if module:
            self.model = self.model.module

    # Training function.
    def train_one_epoch(
        self,
        epoch,
        conf,
        trainloader,
        optimizer,
        criterion,
        scaler,
        scheduler,
        metrics
    ):

        batches_per_epoch = conf['trainer']['batches_per_epoch']
        grad_accum_every = conf['trainer']['grad_accum_every']
        history_len = conf["data"]["history_len"]
        forecast_len = conf["data"]["forecast_len"]
        amp = conf['trainer']['amp']
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        rollout_p = 1.0 if 'stop_rollout' not in conf['trainer'] else conf['trainer']['stop_rollout']

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "lambda":
            scheduler.step()

        # set up a custom tqdm
        if isinstance(trainloader.dataset, IterableDataset):
            # we sample forecast termination with probability p during training
            trainloader.dataset.set_rollout_prob(rollout_p)
        else:
            batches_per_epoch = (
                batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)
            )

        batch_group_generator = tqdm.tqdm(
            enumerate(trainloader),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False
        )

        results_dict = defaultdict(list)

        for i, batch in batch_group_generator:

            logs = {}

            commit_loss = 0.0

            with autocast(enabled=amp):

                x = self.model.concat_and_reshape(
                    batch["x"],
                    batch["x_surf"]
                ).to(self.device)

                y = self.model.concat_and_reshape(
                    batch["y"],
                    batch["y_surf"]
                )

                k = 0
                while True:

                    with torch.no_grad() if k != forecast_len else torch.enable_grad():

                        self.model.eval() if k != forecast_len else self.model.train()

                        if getattr(self.model, 'use_codebook', False):
                            y_pred, cm_loss = self.model(x)
                            commit_loss += cm_loss
                        else:
                            y_pred = self.model(x)

                        if k == forecast_len:
                            break

                        k += 1

                        if history_len > 1:
                            x_detach = x[:, :, 1:].detach()
                            x = torch.cat([x_detach, y_pred.detach()], dim=2).detach()
                            del x_detach
                        else:
                            x = y_pred.detach()

                y = y.to(device=self.device, dtype=y_pred.dtype)
                loss = criterion(y, y_pred)

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                loss = loss.mean() + commit_loss

                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {'loss': loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            if 'forecast_hour' in batch:
                forecast_hour_stop = batch['forecast_hour'][-1].item()
                results_dict["train_forecast_len"].append(forecast_hour_stop+1)
            else:
                results_dict["train_forecast_len"].append(forecast_len+1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
                    raise optuna.TrialPruned()
                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len {:.6}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                np.mean(results_dict["train_forecast_len"])
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "cosine-annealing":
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(
        self,
        epoch,
        conf,
        valid_loader,
        criterion,
        metrics
    ):

        self.model.eval()

        valid_batches_per_epoch = conf['trainer']['valid_batches_per_epoch']
        history_len = conf["data"]["valid_history_len"] if "valid_history_len" in conf["data"] else conf["history_len"]
        forecast_len = conf["data"]["valid_forecast_len"] if "valid_forecast_len" in conf["data"] else conf["forecast_len"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        if isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = valid_batches_per_epoch
        else:
            valid_batches_per_epoch = (
                valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)
            )

        batch_group_generator = tqdm.tqdm(
            enumerate(valid_loader),
            total=valid_batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False
        )

        for i, batch in batch_group_generator:

            with torch.no_grad():

                commit_loss = 0.0

                x = self.model.concat_and_reshape(
                    batch["x"],
                    batch["x_surf"]
                ).to(self.device)

                y = self.model.concat_and_reshape(
                    batch["y"],
                    batch["y_surf"]
                ).to(self.device)

                k = 0
                while True:
                    if getattr(self.model, 'use_codebook', False):
                        y_pred, cm_loss = self.model(x)
                        commit_loss += cm_loss
                    else:
                        y_pred = self.model(x)

                    if k == forecast_len:
                        break

                    k += 1

                    if history_len > 1:
                        x_detach = x[:, :, 1:].detach()
                        x = torch.cat([x_detach, y_pred.detach()], dim=2).detach()
                        del x_detach
                    else:
                        x = y_pred.detach()

                loss = criterion(y.to(y_pred.dtype), y_pred)

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                if distributed:
                    torch.distributed.barrier()
                results_dict["valid_loss"].append(batch_loss[0].item())

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"])
                )
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

                if i >= valid_batches_per_epoch and i > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def fit(
        self,
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        rollout_scheduler=None,
        trial=False
    ):
        save_loc = conf['save_loc']
        start_epoch = conf['trainer']['start_epoch']
        epochs = conf['trainer']['epochs']

        # Reload the results saved in the training csv if continuing to train
        if start_epoch == 0:
            results_dict = defaultdict(list)
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(f"{save_loc}/training_log.csv")
            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        for epoch in range(start_epoch, epochs):

            logging.info(f"Beginning epoch {epoch}")

            if not isinstance(train_loader.dataset, IterableDataset):
                train_loader.sampler.set_epoch(epoch)
            else:
                train_loader.dataset.set_epoch(epoch)
                if rollout_scheduler is not None:
                    conf['trainer']['stop_rollout'] = rollout_scheduler(epoch, epochs)
                    train_loader.dataset.set_rollout_prob(conf['trainer']['stop_rollout'])

            ############
            #
            # Train
            #
            ############

            train_results = self.train_one_epoch(
                epoch,
                conf,
                train_loader,
                optimizer,
                train_criterion,
                scaler,
                scheduler,
                metrics
            )

            ############
            #
            # Validation
            #
            ############

            valid_results = self.validate(
                epoch,
                conf,
                valid_loader,
                valid_criterion,
                metrics
            )

            #################
            #
            # Save results
            #
            #################

            # update the learning rate if epoch-by-epoch updates

            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "plateau":
                scheduler.step(results_dict["valid_acc"][-1])

            # Put things into a results dictionary -> dataframe

            results_dict["epoch"].append(epoch)
            for name in ["loss", "acc", "mae"]:
                results_dict[f"train_{name}"].append(np.mean(train_results[f"train_{name}"]))
                results_dict[f"valid_{name}"].append(np.mean(valid_results[f"valid_{name}"]))
            results_dict['train_forecast_len'].append(np.mean(train_results['train_forecast_len']))
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            df = pd.DataFrame.from_dict(results_dict).reset_index()

            # Save the dataframe to disk

            if trial:
                df.to_csv(
                    os.path.join(f"{save_loc}", "trial_results", f"training_log_{trial.number}.csv"),
                    index=False,
                )
            else:
                df.to_csv(os.path.join(f"{save_loc}", "training_log.csv"), index=False)

            ############
            #
            # Checkpoint
            #
            ############

            if not trial:

                if conf["trainer"]["mode"] != "fsdp":

                    if self.rank == 0:

                        # Save the current model

                        logging.info(f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

                        state_dict = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                            'scaler_state_dict': scaler.state_dict()
                        }
                        torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                else:

                    logging.info(f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

                    # Initialize the checkpoint I/O handler
                    checkpoint_io = TorchFSDPCheckpointIO()

                    # Save model and optimizer checkpoints
                    checkpoint_io.save_unsharded_model(
                        self.model,
                        os.path.join(save_loc, "model_checkpoint.pth"),
                        gather_dtensor=True,
                        use_safetensors=False
                    )
                    checkpoint_io.save_unsharded_optimizer(
                        optimizer,
                        os.path.join(save_loc, "optimizer_checkpoint.pth"),
                        gather_dtensor=True
                    )
                    state_dict = {
                        "epoch": epoch,
                        'scheduler_state_dict': scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                        'scaler_state_dict': scaler.state_dict()
                    }
                    torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))

                # This needs updated!
                # valid_loss = np.mean(valid_results["valid_loss"])
                # # save if this is the best model seen so far
                # if (self.rank == 0) and (np.mean(valid_loss) == min(results_dict["valid_loss"])):
                #     if conf["trainer"]["mode"] == "ddp":
                #         shutil.copy(f"{save_loc}/checkpoint_{self.device}.pt", f"{save_loc}/best_{self.device}.pt")
                #     elif conf["trainer"]["mode"] == "fsdp":
                #         if os.path.exists(f"{save_loc}/best"):
                #             shutil.rmtree(f"{save_loc}/best")
                #         shutil.copytree(f"{save_loc}/checkpoint", f"{save_loc}/best")
                #     else:
                #         shutil.copy(f"{save_loc}/checkpoint.pt", f"{save_loc}/best.pt")

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()

            # Report result to the trial
            if trial:
                trial.report(results_dict["valid_loss"][-1], step=epoch)

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict["valid_loss"])
                if j == min(results_dict["valid_loss"])
            ][0]
            offset = epoch - best_epoch
            if offset >= conf['trainer']['stopping_patience']:
                logging.info(f"Trial {trial.number} is stopping early")
                break

            # Stop training if we get too close to the wall time
            if 'stop_after_epoch' in conf['trainer']:
                if conf['trainer']['stop_after_epoch']:
                    break

        best_epoch = [
            i for i, j in enumerate(results_dict["valid_loss"]) if j == min(results_dict["valid_loss"])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
