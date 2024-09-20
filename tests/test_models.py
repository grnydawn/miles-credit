import os
import yaml

import torch

from credit.models import load_model
from credit.models.unet import SegmentationModel
from credit.models.crossformer import CrossFormer

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])


# todo: use config/*.yml as source files for testing
#
#
#


def test_unet():
    #load config
    config = os.path.join(TEST_FILE_DIR, "data/unet_config.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    
    model = load_model(conf)

    assert isinstance(model, SegmentationModel)

    image_height = conf["model"]["image_height"]
    image_width = conf["model"]["image_width"]
    variables = len(conf["data"]["variables"])
    levels = conf["model"]["levels"]
    frames = conf["model"]["frames"]
    surface_variables = len(conf["data"]["surface_variables"])
    static_variables = len(conf["data"]["static_variables"])

    in_channels = int(variables*levels + surface_variables + static_variables)
    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width)

    y_pred = model(input_tensor)
    
    assert y_pred.shape == torch.Size([1, in_channels, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()

# def test_crossformer(): 
#     # todo: test model loading later
#     # todo: reduce model size in CI

#     image_height = 640  # 640, 192
#     image_width = 1280  # 1280, 288
#     levels = 16
#     frames = 2
#     channels = 4
#     surface_channels = 7
#     input_only_channels = 3
#     frame_patch_size = 2
#     pad_lon = 80
#     pad_lat = 80
#     post_conf={"use_skebs": True, "image_width": image_width}

#     in_channels = channels * levels + surface_channels + input_only_channels
#     input_tensor = torch.randn(1, in_channels, frames, image_height, image_width)

#     model = CrossFormer(
#         image_height=image_height,
#         image_width=image_width,
#         frames=frames,
#         frame_patch_size=frame_patch_size,
#         channels=channels,
#         surface_channels=surface_channels,
#         input_only_channels=input_only_channels,
#         levels=levels,
#         dim=(128, 256, 512, 1024),
#         depth=(2, 2, 18, 2),
#         global_window_size=(8, 4, 2, 1),
#         local_window_size=5,
#         cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
#         cross_embed_strides=(4, 2, 2, 2),
#         attn_dropout=0.,
#         ff_dropout=0.,
#         pad_lon=pad_lon,
#         pad_lat=pad_lat,
#         post_conf=post_conf,
#     )


#     y_pred = model(input_tensor)
    
#     assert y_pred.shape == torch.Size([1, in_channels - input_only_channels, 1, image_height, image_width])
#     assert not torch.isnan(y_pred).any()

if __name__ == "__main__":
    test_unet()
    # test_crossformer()



