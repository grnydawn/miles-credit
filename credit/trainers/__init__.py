
import copy
import logging

# Import trainer classes
from credit.trainers.trainerERA5_v1 import Trainer as TrainerDeprecated
from credit.trainers.trainerERA5_v2 import Trainer as TrainerStandard
from credit.trainers.trainerERA5_multistep import Trainer as TrainerMultiStep
from credit.trainers.trainer404 import Trainer as Trainer404

logger = logging.getLogger(__name__)


# Define trainer types and their corresponding classes
trainer_types = {
    "standard-deprecated": (TrainerDeprecated, "Loading a deprecated version of the standard trainer. It will not work with forcing config options."),
    "standard": (TrainerStandard, "Loading a standard trainer."),
    "multi-step": (TrainerMultiStep, "Loading a multi-step trainer."),
    "conus404": (Trainer404, "Loading a standard trainer for the CONUS404 dataset.")
}


def load_trainer(conf, load_weights=False):
    conf = copy.deepcopy(conf)
    trainer_conf = conf["trainer"]

    if "type" not in trainer_conf:
        msg = f"You need to specify a trainer 'type' in the config file. Choose from {list(trainer_types.keys())}"
        logger.warning(msg)
        raise ValueError(msg)

    trainer_type = trainer_conf.pop("type")

    if trainer_type in trainer_types:
        trainer, message = trainer_types[trainer_type]
        logger.info(message)
        return trainer

    else:
        msg = f"Trainer type {trainer_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
