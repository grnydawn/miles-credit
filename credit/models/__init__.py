import os
import copy
import logging
from importlib.metadata import version

import torch
import torch.nn.functional as F

# Import model classes
from credit.models.crossformer import CrossFormer
from credit.models.crossformer_may1 import CrossFormer as CrossFormerDep
from credit.models.simple_vit import SimpleViT
from credit.models.cube_vit import CubeViT
from credit.models.vit2d import ViT2D
from credit.models.vit3d import ViT3D
from credit.models.rvt import RViT
from credit.models.unet import SegmentationModel
from credit.models.unet404 import SegmentationModel404
from credit.models.fuxi import Fuxi
from credit.models.swin import SwinTransformerV2Cr
from credit.models.graph import GraphResTransfGRU

logger = logging.getLogger(__name__)

# Define model types and their corresponding classes
model_types = {
    "vit": (ViT2D, "Loading a Vision transformer architecture ..."),
    "vit3d": (ViT3D, "Loading a Vision transformer architecture ..."),
    "rvt": (RViT, "Loading a custom rotary transformer architecture with conv attention ..."),
    "simple-vit": (SimpleViT, "Loading a simplified vit rotary transformer architecture ..."),
    "cube-vit": (CubeViT, "Loading a simplified vit rotary transformer architecture with a 3D conv tokenizer ..."),
    "crossformer": (CrossFormer, "Loading the CrossFormer model with a conv decoder head and skip connections ..."),
    "crossformer-deprecated": (CrossFormerDep, "Loading the CrossFormer model with a conv decoder head and skip connections ..."),
    "unet": (SegmentationModel, "Loading a unet model"),
    "unet404": (SegmentationModel404, "Loading unet404 model"),
    "fuxi": (Fuxi, "Loading Fuxi model"),
    "swin": (SwinTransformerV2Cr, "Loading the minimal Swin model"),
    "graph": (GraphResTransfGRU, "Loading Graph Residual Transformer GRU model")
} 


def load_model(conf, load_weights=False):
    conf = copy.deepcopy(conf)
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    if model_type in ('unet404',):
        import torch
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            model = model(conf)
            save_loc = conf['save_loc']
            ckpt = os.path.join(save_loc, "checkpoint.pt")

            if not os.path.isfile(ckpt):
                raise ValueError(
                    "No saved checkpoint exists. You must train a model first. Exiting."
                )

            logging.info(
                f"Loading a model with pre-trained weights from path {ckpt}"
            )

            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
            
        return model(conf)

    if model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            return model.load_model(conf)
        return model(**model_conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)


def earth_padding(x, pad_NS, pad_WE):
    '''
    Apply paddings on poles and longitude boundaries.

    Args:
        x (torch.Tensor): The padded tensor to unpad.
        pad_NS (list[int]): Padding sizes for the North-South (latitude) dimension [top, bottom].
        pad_WE (list[int]): Padding sizes for the West-East (longitude) dimension [left, right].

    Returns:
        torch.Tensor: The unpadded tensor with the original size.
    '''
    
    if any(p > 0 for p in pad_NS):
        # 180-degree shift using half the longitude size
        shift_size = int(x.shape[-1] // 2)
        xroll = torch.roll(x, shifts=shift_size, dims=-1)
    
        # pad poles
        xroll_flip_top = torch.flip(xroll[..., :pad_NS[0], :], (-2,))
        xroll_flip_bot = torch.flip(xroll[..., -pad_NS[1]:, :], (-2,))
        x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)
    
    if any(p > 0 for p in pad_WE):
        x = F.pad(x, (pad_WE[0], pad_WE[1], 0, 0, 0, 0), mode='circular')

    return x

def earth_unpad(x, pad_NS, pad_WE):
    '''
    Removes the padding applied by earth_padding to restore the original tensor size.

    Args:
        x (torch.Tensor): The padded tensor to unpad.
        pad_NS (list[int]): Padding sizes for the North-South (latitude) dimension [top, bottom].
        pad_WE (list[int]): Padding sizes for the West-East (longitude) dimension [left, right].

    Returns:
        torch.Tensor: The unpadded tensor with the original size.
    '''
    # unpad along the lat dim
    if any(p > 0 for p in pad_NS):
        start_NS = pad_NS[0]
        end_NS = -pad_NS[1] if pad_NS[1] > 0 else None
        x = x[..., start_NS:end_NS, :]

    # unpad along the lon dim
    if any(p > 0 for p in pad_WE):
        start_WE = pad_WE[0]
        end_WE = -pad_WE[1] if pad_WE[1] > 0 else None
        x = x[..., :, start_WE:end_WE]

    return x

def mirror_padding(x, pad_NS, pad_WE):
    '''
    Applies padding to the tensor x along latitude and longitude dimensions.
    Uses 'reflect' mode for latitude (north-south) padding and 'circular' mode for longitude (east-west) padding.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, var, time, lat, lon)
        pad_NS (list[int]): [top_pad, bottom_pad] for latitude (north-south)
        pad_WE (list[int]): [left_pad, right_pad] for longitude (east-west)

    Returns:
        torch.Tensor: Padded tensor
    '''
    # Pad along longitude (east-west)
    if any(p > 0 for p in pad_WE):
        pad_lon_left, pad_lon_right = pad_WE
        # Pad the last dimension (longitude)
        x = F.pad(x, pad=(pad_lon_left, pad_lon_right, 0, 0, 0, 0), mode='circular')
    
    # Pad along latitude (north-south)
    if any(p > 0 for p in pad_NS):
        # Reshape to combine var and time dimensions
        x_shape = x.shape  # (batch, var, time, lat, lon)
        x = x.reshape(x_shape[0], x_shape[1] * x_shape[2], x_shape[3], x_shape[4])
        pad_lat_top, pad_lat_bottom = pad_NS
        # Pad the height dimension (latitude)
        x = F.pad(x, pad=(0, 0, pad_lat_top, pad_lat_bottom, 0, 0), mode='reflect')
        # Reshape back to original shape with updated latitude dimension
        new_lat = x_shape[3] + pad_lat_top + pad_lat_bottom
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], new_lat, x_shape[4])
    
    return x

def mirror_unpad(x, pad_NS, pad_WE):
    '''
    Removes the padding applied by mirror_padding to restore the original tensor size.

    Args:
        x (torch.Tensor): The padded tensor to unpad.
        pad_NS (list[int]): [top_pad, bottom_pad] for latitude (north-south).
        pad_WE (list[int]): [left_pad, right_pad] for longitude (east-west).

    Returns:
        torch.Tensor: The unpadded tensor with the original size.
    '''
    # Unpad along latitude (north-south)
    if any(p > 0 for p in pad_NS):
        pad_lat_top, pad_lat_bottom = pad_NS
        start_NS = pad_lat_top
        end_NS = -pad_lat_bottom if pad_lat_bottom > 0 else None
        x_shape = x.shape
        # Reshape to combine var and time dimensions
        x = x.reshape(x_shape[0], x_shape[1] * x_shape[2], x_shape[3], x_shape[4])
        # Slice along the latitude dimension
        x = x[..., start_NS:end_NS, :]
        # Reshape back to original shape with updated latitude dimension
        new_lat = x.shape[2]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], new_lat, x_shape[4])

    # Unpad along longitude (east-west)
    if any(p > 0 for p in pad_WE):
        pad_lon_left, pad_lon_right = pad_WE
        start_WE = pad_lon_left
        end_WE = -pad_lon_right if pad_lon_right > 0 else None
        # Slice along the longitude dimension
        x = x[..., :, start_WE:end_WE]

    return x
# dont need an old timm version anymore https://github.com/qubvel/segmentation_models.pytorch/releases/tag/v0.3.3
# def check_timm_version(model_type):
#     if model_type == "unet":
#         try:
#             assert (version('timm') == '0.6.12')
#         except AssertionError as e:
#             msg = """timm version 0.6 is required for using pytorch-segmentation-models. Please use environment-unet.yml env or pip install timm==0.6.12."""
#             raise Exception(msg) from e
#     elif model_type == "fuxi":
#         try:
#             assert (version('timm') >= '0.9.12')
#         except AssertionError as e:
#             msg = """timm version 0.9.12 or greater is required for FuXi model. Please use environment.yml env or pip install timm==0.9.12."""
#             raise Exception(msg) from e
