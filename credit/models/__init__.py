from credit.models.unet import SegmentationModel
from credit.models.vit2d import ViT2D
from credit.models.rvt import RViT
import logging


logger = logging.getLogger(__name__)


def load_model(conf):
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf["type"]
    del model_conf["type"]

    if model_type == "vit":
        logger.info("Loading a Vision transformer architecture ...")
        return ViT2D(**model_conf)

    elif model_type == "rvt":
        logger.info("Loading a custom rotary transformer architecture with conv attention ...")
        return RViT(**model_conf)

    elif model_type == "unet":
        logger.info("Loading a segmentation model ...")
        return SegmentationModel(conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
