'''
base_trainer.py
-------------------------------------------------------
Content:
    - Trainer
        - train_one_epoch
        - validate
        - fit
'''
import os
import gc
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import IterableDataset
from credit.models.checkpoint import TorchFSDPCheckpointIO
from credit.scheduler import update_on_epoch
from credit.trainers.utils import cleanup

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):

    def __init__(self, model: torch.nn.Module, rank: int, module: bool = False):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            rank (int): The rank of the current process.
            module (bool): Whether to use model.module instead of model directly.
        """
        super(BaseTrainer, self).__init__()
        self.model = model.module if module else model
        self.rank = rank
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

    @abstractmethod
    def train_one_epoch(
        self,
        epoch: int,
        conf: Dict[str, Any],
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            conf (Dict[str, Any]): The configuration dictionary.
            trainloader (torch.utils.data.DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (torch.nn.Module): The loss function.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            metrics (Dict[str, Any]): The metrics to track during training.

        Returns:
            Dict[str, float]: A dictionary containing the training results.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        epoch: int,
        conf: Dict[str, Any],
        valid_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Args:
            epoch (int): The current epoch number.
            conf (Dict[str, Any]): The configuration dictionary.
            valid_loader (torch.utils.data.DataLoader): The validation data loader.
            criterion (torch.nn.Module): The loss function.
            metrics (Dict[str, Any]): The metrics to track during validation.

        Returns:
            Dict[str, float]: A dictionary containing the validation results.
        """
        raise NotImplementedError

    def save_checkpoint(self, 
                        epoch: int,
                        optimizer: torch.optim.Optimizer,
                        scaler: torch.cuda.amp.GradScaler,
                        scheduler, #: torch.optim.lr_scheduler._LRScheduler or None
                        save_loc: str,
                        state_dict: Dict[str, Any],
                        prefix: str = None) -> None:
        """
        Save a checkpoint of the model.

        Args:
            epoch (int): The current epoch number.
            optimizer (torch.optim.Optimizer): The optimizer.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler or None if no scheduler applied.
            save_loc (str): The location to save the checkpoint.
            state_dict (Dict[str, Any]): The state dictionary to save.
            prefix (str): prefix of the file names, None will save checkpoint.pt
        """
        if scheduler is None:
            scheduler_state_dict = None
        else:
            scheduler_state_dict = scheduler.state_dict()
        
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dict,
            'scaler_state_dict': scaler.state_dict()
        }
        
        if prefix is None:
            torch.save(state_dict, f"{save_loc}/checkpoint.pt")
            logger.info(f"Saved checkpoint to {save_loc}/checkpoint.pt")
        else:
            torch.save(state_dict, f"{save_loc}/{prefix}_checkpoint.pt")
            logger.info(f"Saved checkpoint to {save_loc}/{prefix}_checkpoint.pt")
            
    def save_fsdp_checkpoint(self,
                             epoch: int,
                             optimizer: torch.optim.Optimizer,
                             scaler: torch.cuda.amp.GradScaler,
                             scheduler, #: torch.optim.lr_scheduler._LRScheduler or None
                             save_loc: str, 
                             state_dict: Dict[str, Any],
                             prefix: str = None) -> None:
        """
        Save a checkpoint for FSDP training.

        Args:
            epoch (int): The current epoch number.
            optimizer (torch.optim.Optimizer): The optimizer.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler or None if no scheduler applied.
            save_loc (str): The location to save the checkpoint.
            state_dict (Dict[str, Any]): The state dictionary to save.
            prefix (str): prefix of the file names, None will save checkpoint.pt, model_checkpoint.pt, optimizer_checkpoint.pt
        """
        if prefix is None:
            save_loc_model = os.path.join(save_loc, "model_checkpoint.pt")
            save_loc_optm = os.path.join(save_loc, "optimizer_checkpoint.pt")
            save_loc_checkpoint = os.path.join(save_loc, "checkpoint.pt")
        else:
            save_loc_model = os.path.join(save_loc, f"{prefix}_model_checkpoint.pt")
            save_loc_optm = os.path.join(save_loc, f"{prefix}_optimizer_checkpoint.pt")
            save_loc_checkpoint = os.path.join(save_loc, f"{prefix}_checkpoint.pt")
        
        if scheduler is None:
            scheduler_state_dict = None
        else:
            scheduler_state_dict = scheduler.state_dict()
            
        # Initialize the checkpoint I/O handler
        checkpoint_io = TorchFSDPCheckpointIO()
        
        # Save model and optimizer checkpoints
        checkpoint_io.save_unsharded_model(
            self.model,
            save_loc_model,
            gather_dtensor=True,
            use_safetensors=False,
            rank=self.rank
        )
        logger.info(f"Saved model checkpoint to {save_loc_model}")
        
        checkpoint_io.save_unsharded_optimizer(
            optimizer,
            save_loc_optm,
            gather_dtensor=True,
            rank=self.rank
        )
        logger.info(f"Saved optm checkpoint to {save_loc_optm}")
        
        # Still need to save the scheduler and scaler states, just in another file for FSDP
        state_dict = {
            "epoch": epoch,
            'scheduler_state_dict': scheduler_state_dict,
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(state_dict, save_loc_checkpoint)
        
        logger.info(f"Saved state_dict to {save_loc_checkpoint}")
        

    def fit(
        self,
        conf: Dict[str, Any],
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: Optimizer,
        train_criterion: torch.nn.Module,
        valid_criterion: torch.nn.Module,
        scaler: GradScaler,
        scheduler: _LRScheduler,
        metrics: Dict[str, Any],
        rollout_scheduler: Optional[callable] = None,
        trial: bool = False
    ) -> Dict[str, Any]:

        """
        Fit the model to the data.

        Args:
            conf (Dict[str, Any]): Configuration dictionary.
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): The optimizer to use for training.
            train_criterion (torch.nn.Module): Loss function for training.
            valid_criterion (torch.nn.Module): Loss function for validation.
            scaler (GradScaler): Gradient scaler for mixed precision training.
            scheduler (_LRScheduler): Learning rate scheduler.
            metrics (Dict[str, Any]): Dictionary of metrics to track during training.
            rollout_scheduler (Optional[callable]): Function to schedule rollout probability, if applicable.
            trial (bool): Whether this is a trial run (e.g., for hyperparameter tuning).

        Returns:
            Dict[str, Any]: Dictionary containing the best results from training.
        """

        # convert $USER to the actual user name
        conf['save_loc'] = save_loc = os.path.expandvars(conf['save_loc'])

        # training hyperparameters
        start_epoch = conf['trainer']['start_epoch']
        epochs = conf['trainer']['epochs']
        skip_validation = conf['trainer']['skip_validation'] if 'skip_validation' in conf['trainer'] else False
        flag_load_weights = conf['trainer']['load_weights']

        # =========================================== #
        # user can specify to run a fixed number of epochs
        if 'num_epoch' in conf['trainer']:
            print('the current job will run {} epochs max'.format(conf['trainer']['num_epoch']))
        else:
            conf['trainer']['num_epoch'] = 9999
        # =========================================== #
        
        # Reload the results saved in the training csv if continuing to train
        if (start_epoch == 0) or (flag_load_weights is False):
            results_dict = defaultdict(list)
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(os.path.join(save_loc, "training_log.csv"))

            # Set start_epoch to the length of the training log and train for one epoch
            # This is a manual override, you must use train_one_epoch = True
            if "train_one_epoch" in conf["trainer"] and conf["trainer"]["train_one_epoch"]:
                start_epoch = len(saved_results)
                epochs = start_epoch + 1

            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])
                
        count = 0
        for epoch in range(start_epoch, epochs):

            if count >= conf['trainer']['num_epoch']:
                print('{} epochs completed, exit'.format(conf['trainer']['num_epoch']))
                break;
            
            logging.info(f"Beginning epoch {epoch}")

            if not isinstance(train_loader.dataset, IterableDataset):
                train_loader.sampler.set_epoch(epoch)
            else:
                train_loader.dataset.set_epoch(epoch)

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

            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] in update_on_epoch:
                flag_use_scheduler = True
                if conf['trainer']['scheduler']['scheduler_type'] == 'plateau':
                    scheduler.step(results_dict["valid_acc"][-1])
                else:
                    scheduler.step()
            else:
                flag_use_scheduler = False

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
            
            # ================================================================================================================= #
            # Save the current epoch
            if not trial:
                # non-fsdp check-pointing
                if conf["trainer"]["mode"] != "fsdp":
                    if self.rank == 0:
                        # logging.info(f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")
                        if flag_use_scheduler:
                            save_checkpoint(epoch, optimizer, scaler, scheduler, save_loc, state_dict, prefix=None)
                        else:
                            save_checkpoint(epoch, optimizer, scaler, None, save_loc, state_dict, prefix=None)
                else:
                    # fsdp check-pointing
                    # logging.info(f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")
                    if flag_use_scheduler:
                        save_fsdp_checkpoint(epoch, optimizer, scaler, scheduler, save_loc, state_dict, prefix=None)
                    else:
                        save_fsdp_checkpoint(epoch, optimizer, scaler, None, save_loc, state_dict, prefix=None)
                    
            # ================================================================================================================= #

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

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [i for i, j in enumerate(results_dict[training_metric]) if j == min(results_dict[training_metric])][0]
            offset = epoch - best_epoch

            # if the current epoch is the best epoch, save the "best" checkpoint 
            if offset == 0:
                # ================================================================================================================= #
                # Save the best checkpoint
                if not trial:
                    # non-fsdp check-pointing
                    if conf["trainer"]["mode"] != "fsdp":
                        if self.rank == 0:
                            if flag_use_scheduler:
                                save_checkpoint(epoch, optimizer, scaler, scheduler, save_loc, state_dict, prefix='best')
                            else:
                                save_checkpoint(epoch, optimizer, scaler, None, save_loc, state_dict, prefix='best')
                    else:
                        # fsdp check-pointing
                        if flag_use_scheduler:
                            save_fsdp_checkpoint(epoch, optimizer, scaler, scheduler, save_loc, state_dict, prefix='best')
                        else:
                            save_fsdp_checkpoint(epoch, optimizer, scaler, None, save_loc, state_dict, prefix='best')
                # ================================================================================================================= #
            
            # early stopping
            if offset >= conf['trainer']['stopping_patience']:
                logging.info("Best validation set scores were found in epoch {}; current epoch is {}; early stopping triigered".format(
                    best_epoch, epoch))
                break

            count += 1
            
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
