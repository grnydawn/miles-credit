import gc
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
import optuna
from credit.models.checkpoint import TorchFSDPCheckpointIO
from glob import glob
from credit.transforms import load_transforms
from credit.data import ERA5Dataset


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


class TOADataLoader:
    # This should get moved to solar.py at some point
    def __init__(self, conf):
        self.TOA = xr.open_dataset(conf["data"]["TOA_forcing_path"])
        self.times_b = pd.to_datetime(self.TOA.time.values)

    def __call__(self, datetime_input):
        doy = datetime_input.dayofyear
        hod = datetime_input.hour
        mask_toa = [doy == time.dayofyear and hod == time.hour for time in self.times_b]
        return torch.tensor(((self.TOA['tsi'].sel(time=mask_toa))/2540585.74).to_numpy()).unsqueeze(0)


class ReplayBuffer:
    def __init__(self, conf, buffer_size=32, device="cpu", dtype=np.float32, rank=0):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.dtype = dtype
        self.device = device
        self.rank = rank

        # Extract relevant parameters from conf
        data_conf = conf['data']
        filenames = data_conf.get('save_loc')
        history_len = data_conf.get('history_len', 2)
        forecast_len = data_conf.get('forecast_len', 0)
        transform = data_conf.get('transform', None)

        model_conf = conf['model']
        input_shape = (
            model_conf['levels'] * model_conf['channels'] + model_conf['surface_channels'] + model_conf['static_channels'],
            model_conf['frames'],
            model_conf['image_height'],
            model_conf['image_width']
        )

        self.input_shape = input_shape

        # Initialize forecast hour and index buffers
        self.forecast_hour = np.zeros((buffer_size,), dtype=np.int32)
        self.index = np.zeros((buffer_size,), dtype=np.int32)

        # File names
        filenames = sorted(glob(filenames))

        # Preprocessing transformations
        transform = load_transforms(conf)

        # Initialize dataset
        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            transform=transform
        )

        # Create a directory to store numpy files
        self.numpy_dir = os.path.join(conf["save_loc"], "buffer")
        os.makedirs(self.numpy_dir, exist_ok=True)

        # Reload if the dataset exists
        self.reload()

    def add(self, x, lookup_key):
        """Add new experience to the buffer."""
        if self.size < self.buffer_size:
            file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{self.ptr}.npy")
            np.save(file_path, x.cpu().numpy())
            self.index[self.ptr] = lookup_key.item()
            self.forecast_hour[self.ptr] = 1  # Initialize forecast_hour to 1
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size += 1
        else:
            # Replace the entry with the smallest forecast hour if buffer is full
            min_forecast_hour_idx = np.argmin(self.forecast_hour)
            file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{min_forecast_hour_idx}.npy")
            np.save(file_path, x.cpu().numpy())
            self.index[min_forecast_hour_idx] = lookup_key.item()
            self.forecast_hour[min_forecast_hour_idx] = 1  # Reset forecast hour

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer, increment forecast_hour, and update x with new predictions."""
        weights = self.forecast_hour[:self.size] / np.sum(self.forecast_hour[:self.size])
        indices = np.random.choice(self.size, batch_size, replace=False, p=weights)

        # Increment forecast_hour for sampled indices
        self.forecast_hour[indices] += 1

        x_batch = np.zeros((batch_size, *self.input_shape), dtype=self.dtype)

        for i, idx in enumerate(indices):
            file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{idx}.npy")
            x_batch[i] = np.load(file_path, mmap_mode='r')

        x_batch = torch.FloatTensor(x_batch).to(self.device)
        return indices, x_batch

    def update(self, indices, new_x, new_lookup_key):
        """Update existing data in the buffer."""
        for i, idx in enumerate(indices):
            file_path = os.path.join(self.numpy_dir, f"buffer_{self.rank}_{idx}.npy")
            np.save(file_path, new_x[i].cpu().numpy())
            self.index[idx] = new_lookup_key[i]

    def update_with_predictions(self, model, sample_size):
        """Use stored predictions as inputs for future predictions."""
        indices, x_sample = self.sample(sample_size)
        ave_forecast_len = np.mean([t-1 for t in self.forecast_hour[indices]])
    
        # Predict using the model
        y_predict = model(x_sample)
    
        y_truth = []
        x_update = []
    
        for i, idx in enumerate(indices):
            lookup_key = self.index[idx]
            
            # Load the next set of inputs using the lookup_key
            y, _ = self.load_inputs(lookup_key + 1)
    
            static = y[:, 67:]
            y_non_static = y[:, :67, 1:]
    
            y_truth.append(y_non_static)
    
            y_pred = y_predict[i].unsqueeze(0).cpu().detach()
            y_pred = torch.cat((y_pred, static[:, :, 1:2, :, :].cpu()), dim=1)
            y_pred = torch.cat([x_sample[i:i+1, :, 1:2, :, :].cpu(), y_pred], dim=2)
            x_update.append(y_pred)
    
        x_update = torch.cat(x_update, dim=0)
        y_truth = torch.cat(y_truth, dim=0)
    
        new_lookup_keys = self.index[indices] + 1
        self.update(indices, x_update, new_lookup_keys)
        self.save()
    
        return y_predict, y_truth, ave_forecast_len

    def concat_and_reshape(self, x1, x2):
        x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def load_inputs(self, idx):
        sample = self.dataset.__getitem__(idx)

        x = self.concat_and_reshape(
                sample["x"].unsqueeze(0),
                sample["x_surf"].unsqueeze(0)
        )

        if "static" in sample:
            static = torch.FloatTensor(sample["static"]).unsqueeze(0).expand(2, -1, -1, -1)
            x = torch.cat([x, static.unsqueeze(0)], dim=1)

        if "TOA" in sample:
            toa = torch.FloatTensor(sample["TOA"]).unsqueeze(0)
            x = torch.cat([x, toa.unsqueeze(1)], dim=1)

        y = self.concat_and_reshape(
            sample["y"].unsqueeze(0),
            sample["y_surf"].unsqueeze(0)
        )

        return x, y

    def populate(self):
        """Populate the buffer with random data points from the dataset."""
        dataset_size = len(self.dataset)
        random_indices = np.random.choice(dataset_size, self.buffer_size, replace=False)

        for i, idx in tqdm.tqdm(enumerate(random_indices), total=len(random_indices)):
            x, _ = self.load_inputs(idx)
            self.add(x, idx)

        self.size = self.buffer_size

    def save(self):
        """Save the forecast hours, index arrays, pointer, and size to disk."""
        np.save(os.path.join(self.numpy_dir, f'forecast_hours_{self.rank}.npy'), self.forecast_hour)
        np.save(os.path.join(self.numpy_dir, f'index_{self.rank}.npy'), self.index)
        np.save(os.path.join(self.numpy_dir, f'ptr_{self.rank}.npy'), np.array([self.ptr]))
        np.save(os.path.join(self.numpy_dir, f'size_{self.rank}.npy'), np.array([self.size]))

    def reload(self):
        """Reload the buffer from saved numpy files."""
        forecast_hour_path = os.path.join(self.numpy_dir, f'forecast_hours_{self.rank}.npy')
        index_path = os.path.join(self.numpy_dir, f'index_{self.rank}.npy')
        ptr_path = os.path.join(self.numpy_dir, f'ptr_{self.rank}.npy')
        size_path = os.path.join(self.numpy_dir, f'size_{self.rank}.npy')
    
        if os.path.exists(forecast_hour_path):
            self.forecast_hour = np.load(forecast_hour_path)
        else:
            self.forecast_hour = np.zeros((self.buffer_size,), dtype=np.int32)
    
        if os.path.exists(index_path):
            self.index = np.load(index_path)
        else:
            self.index = np.zeros((self.buffer_size,), dtype=np.int32)
    
        if os.path.exists(ptr_path):
            self.ptr = np.load(ptr_path)[0]
        else:
            self.ptr = 0
    
        if os.path.exists(size_path):
            self.size = np.load(size_path)[0]
        else:
            self.size = 0


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
        total_time_steps = conf["data"]["total_time_steps"] if "total_time_steps" in conf["data"] else forecast_len
        batch_size = conf["trainer"]["train_batch_size"]

        if "static_variables" in conf["data"] and "tsi" in conf["data"]["static_variables"]:
            self.toa = TOADataLoader(conf)

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

        static = None
        results_dict = defaultdict(list)

        replay_buffer = ReplayBuffer(
            conf,
            device=self.device,
            rank=self.rank,
            buffer_size=100
        )

        self.model.train()

        for i, batch in batch_group_generator:

            logs = {}

            commit_loss = 0.0

            with autocast(enabled=amp):

                x = self.model.concat_and_reshape(
                    batch["x"],
                    batch["x_surf"]
                ).to(self.device)

                if "static" in batch:
                    if static is None:
                        static = batch["static"].to(self.device).unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1).float()
                    x = torch.cat((x, static.clone()), dim=1)

                if "TOA" in batch:
                    toa = batch["TOA"].to(self.device)
                    x = torch.cat([x, toa.unsqueeze(1)], dim=1)

                replay_buffer.add(x, batch["index"])
                y_pred, y, ave_forecast_hour = replay_buffer.update_with_predictions(self.model, batch_size)

                # sample from the buffer
                y = y.to(self.device)
                y_pred = y_pred.to(self.device)
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
            results_dict["train_forecast_len"].append(ave_forecast_hour)

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
        total_time_steps = conf["data"]["total_time_steps"] if "total_time_steps" in conf["data"] else forecast_len

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

        static = None

        for i, batch in batch_group_generator:

            with torch.no_grad():

                commit_loss = 0.0

                x = self.model.concat_and_reshape(
                    batch["x"],
                    batch["x_surf"]
                ).to(self.device)

                if "static" in batch:
                    if static is None:
                        static = batch["static"].to(self.device).unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1).float()
                    x = torch.cat((x, static.clone()), dim=1)

                if "TOA" in batch:
                    toa = batch["TOA"].to(self.device)
                    x = torch.cat([x, toa.unsqueeze(1)], dim=1)

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

                    if k == total_time_steps:
                        break

                    k += 1

                    if history_len > 1:
                        x_detach = x.detach()[:, :, 1:]
                        if "static" in batch:
                            y_pred = torch.cat((y_pred, static[:, :, 0:1].clone()), dim=1)
                        if "TOA" in batch:  # update the TOA based on doy and hod
                            elapsed_time = pd.Timedelta(hours=k)
                            current_times = [pd.to_datetime(_t, unit="ns") + elapsed_time for _t in batch["datetime"]]
                            toa = torch.cat([self.toa(_t).unsqueeze(0) for _t in current_times], dim=0).to(self.device)
                            y_pred = torch.cat([y_pred, toa], dim=1)
                        x = torch.cat([x_detach, y_pred], dim=2).detach()
                    else:
                        if "static" in batch or "TOA" in batch:
                            x = y_pred.detach()
                            if "static" in batch:
                                x = torch.cat((x, static[:, :, 0:1].clone()), dim=1)
                            if "TOA" in batch:  # update the TOA based on doy and hod
                                elapsed_time = pd.Timedelta(hours=k)
                                current_times = [pd.to_datetime(_t, unit="ns") + elapsed_time for _t in batch["datetime"]]
                                toa = torch.cat([self.toa(_t).unsqueeze(0) for _t in current_times], dim=0).to(self.device)
                                x = torch.cat([x, toa], dim=1)
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
        skip_validation = conf['trainer']['skip_validation'] if 'skip_validation' in conf['trainer'] else False

        # Reload the results saved in the training csv if continuing to train
        if start_epoch == 0:
            results_dict = defaultdict(list)
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(os.path.join(save_loc, "training_log.csv"))
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

            if skip_validation:

                valid_results = train_results

            else:

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
                        os.path.join(save_loc, "model_checkpoint.pt"),
                        gather_dtensor=True,
                        use_safetensors=False,
                        rank=self.rank
                    )
                    checkpoint_io.save_unsharded_optimizer(
                        optimizer,
                        os.path.join(save_loc, "optimizer_checkpoint.pt"),
                        gather_dtensor=True,
                        rank=self.rank
                    )

                    # Still need to save the scheduler and scaler states, just in another file for FSDP

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

            training_metric = "train_loss" if skip_validation else "valid_loss"

            # Report result to the trial
            if trial:
                trial.report(results_dict[training_metric][-1], step=epoch)

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict[training_metric])
                if j == min(results_dict[training_metric])
            ][0]
            offset = epoch - best_epoch
            if offset >= conf['trainer']['stopping_patience']:
                logging.info(f"Trial {trial.number} is stopping early")
                break

            # Stop training if we get too close to the wall time
            if 'stop_after_epoch' in conf['trainer']:
                if conf['trainer']['stop_after_epoch']:
                    break

        training_metric = "train_loss" if skip_validation else "valid_loss"

        best_epoch = [
            i for i, j in enumerate(results_dict[training_metric]) if j == min(results_dict[training_metric])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
