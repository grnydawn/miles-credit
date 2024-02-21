import logging

# Import model classes
from credit.models.crossformer import CrossFormer
from credit.models.simple_vit import SimpleViT
from credit.models.cube_vit import CubeViT
from credit.models.vit2d import ViT2D
from credit.models.vit3d import ViT3D
from credit.models.fuxi import Fuxi
from credit.models.rvt import RViT

logger = logging.getLogger(__name__)

# Define model types and their corresponding classes
model_types = {
    "vit": (ViT2D, "Loading a Vision transformer architecture ..."),
    "vit3d": (ViT3D, "Loading a Vision transformer architecture ..."),
    "rvt": (RViT, "Loading a custom rotary transformer architecture with conv attention ..."),
    "simple-vit": (SimpleViT, "Loading a simplified vit rotary transformer architecture ..."),
    "cube-vit": (CubeViT, "Loading a simplified vit rotary transformer architecture with a 3D conv tokenizer ..."),
    "fuxi": (Fuxi, "Loading the FuXi model ..."),
    "crossformer": (CrossFormer, "Loading the CrossFormer model ...")
}


def load_model(conf):
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    if model_type == "unet":
        logger.info("Loading a segmentation model ...")
        try:
            from credit.models.unet import SegmentationModel
            return SegmentationModel(conf)
        except ModuleNotFoundError as E:
            msg = "timm version 6 is required for using pytorch-segmentation-models. Please pip install timm==0.6.12."
            raise ImportError(E)

    elif model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        return model(**model_conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
