import copy
import logging
from importlib.metadata import version

# Import model classes
from credit.models.crossformer import CrossFormer
from credit.models.simple_vit import SimpleViT
from credit.models.cube_vit import CubeViT
from credit.models.vit2d import ViT2D
from credit.models.vit3d import ViT3D
from credit.models.rvt import RViT

logger = logging.getLogger(__name__)

# Define model types and their corresponding classes
model_types = {
    "vit": (ViT2D, "Loading a Vision transformer architecture ..."),
    "vit3d": (ViT3D, "Loading a Vision transformer architecture ..."),
    "rvt": (RViT, "Loading a custom rotary transformer architecture with conv attention ..."),
    "simple-vit": (SimpleViT, "Loading a simplified vit rotary transformer architecture ..."),
    "cube-vit": (CubeViT, "Loading a simplified vit rotary transformer architecture with a 3D conv tokenizer ..."),
    "crossformer": (CrossFormer, "Loading the CrossFormer model with a conv decoder head and skip connections ..."),
}


def load_model(conf, load_weights=False):
    conf = copy.deepcopy(conf)
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    if model_type == "unet":
        logger.info("Loading a segmentation model ...")
        check_timm_version(model_type)
        from credit.models.unet import SegmentationModel
        if load_weights:
            return SegmentationModel.load_model(conf)
        return SegmentationModel(conf)

    elif model_type == "fuxi":
        logger.info("Loading FuXi ...")
        check_timm_version(model_type)

        from credit.models.fuxi import Fuxi
        if load_weights:
            return Fuxi.load_model(conf)
        return Fuxi(**model_conf)

    elif model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            return model.load_model(conf)
        return model(**model_conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)


def check_timm_version(model_type):
    if model_type == "unet":
        try:
            assert (version('timm') == '0.6.12')
        except AssertionError as e:
            msg = """timm version 0.6 is required for using pytorch-segmentation-models. Please use environment-unet.yml env or pip install timm==0.6.12."""
            raise Exception(msg) from e
    elif model_type == "fuxi":
        try:
            assert (version('timm') >= '0.9.12')
        except AssertionError as e:
            msg = """timm version 0.9.12 or greater is required for FuXi model. Please use environment.yml env or pip install timm==0.9.12."""
            raise Exception(msg) from e
