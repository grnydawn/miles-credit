import torch
import torch.nn as nn
from credit.models.crossformer import CrossFormer
from credit.diffusion import GaussianDiffusion
from credit.diffusion import ModifiedGaussianDiffusion

from credit.diffusion import *
import torch.nn.functional as F
import random
from einops import rearrange, reduce
from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple
import logging
import sys


logger = logging.getLogger(__name__)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class CrossFormerDiffusion(CrossFormer):
    def __init__(self, self_condition, condition, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_out_dim = kwargs.get("surface_channels") + (
            kwargs.get("channels") * kwargs.get("levels")
        )  ## channels=total number of output vars out of the wxformer when input+condition is in line.

        self.dim = kwargs.get("dim", (64, 128, 256, 512))  # Default value as in CrossFormer
        self.self_condition = self_condition
        self.condition = condition

        if self.self_condition and self.condition:
            logging.warning("Both self-conditioning and standard conditioning on the manifold via x are not simultanously supported. Exiting")
            sys.exit(0)

        # Adding timestep embedding layer for diffusion
        self.time_mlp = nn.Sequential(nn.Linear(1, self.dim[0]), nn.SiLU(), nn.Linear(self.dim[0], self.dim[-1]))

        self.final_conv = nn.Conv3d(
            self.pre_out_dim, self.output_channels, 1
        )  # used to ensure that only noise channels are left at the end; channels=total number of output vars.

    def forward(self, x, timestep, x_self_cond=False, x_cond=None):
        x_copy = None

        if self.self_condition:
            # input_channels = self.channels * self.levels + self.surface_channels + self.input_only_channels
            # input_channels = self.output_channels
            x_self_cond = default(
                x_self_cond,
                torch.zeros(x.shape[0], self.output_channels, x.shape[2], x.shape[3], x.shape[4], device=x.device)
            )
            x = torch.cat((x_self_cond[:, :self.output_channels], x), dim=1)

        if self.condition:
            x = torch.cat([x, x_cond], dim = 1)

        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:
            x = x.squeeze(2)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        # Add timestep embedding to the feature maps
        t_embed = self.time_mlp(timestep.view(-1, 1).float())  # (B, dim[0])
        t_embed = t_embed[:, :, None, None]  # Reshape to (B, dim[0], 1, 1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])  # Expand to (B, dim[0], H, W)
        x = x + t_embed

        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)
        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)
        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)
        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        x = x.unsqueeze(2)

        x = self.final_conv(x)

        if self.use_post_block:
            x = {"y_pred": x, "x": x_copy}
            x = self.postblock(x)

        return x


def create_model(config, self_condition=True):
    """Initialize and return the CrossFormer model using a config dictionary."""
    return CrossFormerDiffusion(**config).to("cuda")


def create_diffusion(model, config):
    """Initialize and return the Gaussian Diffusion process."""
    return ModifiedGaussianDiffusion(model, **config)


if __name__ == "__main__":
    crossformer_config = {
        "type": "crossformer",
        "frames": 1,  # Number of input states
        "image_height": 192,  # Number of latitude grids
        "image_width": 288,  # Number of longitude grids
        "levels": 16,  # Number of upper-air variable levels
        "channels": 4,  # Upper-air variable channels
        "surface_channels": 7,  # Surface variable channels
        "input_only_channels": 3,  # Dynamic forcing, forcing, static channels
        "output_only_channels": 0,  # Diagnostic variable channels
        "patch_width": 1,  # Number of latitude grids in each 3D patch
        "patch_height": 1,  # Number of longitude grids in each 3D patch
        "frame_patch_size": 1,  # Number of input states in each 3D patch
        "dim": [32, 64, 128, 256],  # Dimensionality of each layer
        "depth": [2, 2, 2, 2],  # Depth of each layer
        "global_window_size": [8, 4, 2, 1],  # Global window size for each layer
        "local_window_size": 8,  # Local window size
        "cross_embed_kernel_sizes": [  # Kernel sizes for cross-embedding
            [4, 8, 16, 32],
            [2, 4],
            [2, 4],
            [2, 4],
        ],
        "cross_embed_strides": [2, 2, 2, 2],  # Strides for cross-embedding
        "attn_dropout": 0.0,  # Dropout probability for attention layers
        "ff_dropout": 0.0,  # Dropout probability for feed-forward layers
        "use_spectral_norm": True,  # Whether to use spectral normalization
        "padding_conf": {
            "activate": True,
            "mode": "earth",
            "pad_lat": [32, 32],
            "pad_lon": [48, 48],
        },
        "interp": False,
        "self_condition": True,
        "pretrained_weights": "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/model_levels/single_step/checkpoint.pt",
    }

    diffusion_config = {
        "image_size": (192, 288),
        "timesteps": 1000,
        "sampling_timesteps": None,
        "objective": "pred_v",
        "beta_schedule": "linear",
        "schedule_fn_kwargs": dict(),
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
        "offset_noise_strength": 0.0,
        "min_snr_loss_weight": False,
        "min_snr_gamma": 5,
        "immiscible": False,
    }

    model = create_model(crossformer_config).to("cuda")
    diffusion = create_diffusion(model, diffusion_config).to("cuda")

    input_tensor = torch.randn(
        1,
        crossformer_config["channels"] * crossformer_config["levels"]
        + crossformer_config["surface_channels"]
        + crossformer_config["input_only_channels"],
        crossformer_config["frames"],
        crossformer_config["image_height"],
        crossformer_config["image_width"],
    ).to("cuda")

    print(input_tensor.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    loss = diffusion(input_tensor).item()  # lazy test

    print("Loss:", loss)

    sampled_images = diffusion.sample(batch_size=1)

    print("Predicted shape:", sampled_images.shape)

    # Extract the last color channel (index -1 for the last channel)
    last_channel = sampled_images[0, -1, 0, :, :]

    import matplotlib.pyplot as plt

    # Plot and save the image
    plt.imshow(last_channel.cpu().numpy(), cmap="gray")  # Display in grayscale
    plt.axis("off")  # Turn off the axis
    plt.savefig("last_channel.png", bbox_inches="tight", pad_inches=0)
    plt.close()
