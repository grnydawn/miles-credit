import segmentation_models_pytorch as smp
import torch
import logging
import copy
import os
import torch.distributed.checkpoint as DCP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = logging.getLogger(__name__)


supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus
}

supported_encoders = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4',
                      'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l']


def load_model(model_conf):
    model_conf = copy.deepcopy(model_conf)
    name = model_conf.pop("name")
    if name in supported_models:
        logger.info(f"Loading model {name} with settings {model_conf}")
        return supported_models[name](**model_conf)
    else:
        raise OSError(
            f"Model name {name} not recognized. Please choose from {supported_models.keys()}")


class SegmentationModel(torch.nn.Module):

    def __init__(self, conf):

        super(SegmentationModel, self).__init__()

        self.num_atmos_vars = conf["model"]["channels"]
        self.num_levels = conf["model"]["frames"]
        self.num_single_layer = conf["model"]["surface_channels"]
        self.use_codebook = False
        self.rk4_integration = conf["model"]["rk4_integration"]
        self.channels = 1

        in_out_channels = int(self.num_atmos_vars*self.num_levels + self.num_single_layer)

        if conf['model']['architecture']['name'] == 'unet':
            conf['model']['architecture']['decoder_attention_type'] = 'scse'
        conf['model']['architecture']['in_channels'] = in_out_channels
        conf['model']['architecture']['classes'] = in_out_channels

        self.model = load_model(conf['model']['architecture'])

    def forward(self, x):
        return self.rk4(x) if self.rk4_integration else self.model(x)

    def concat_and_reshape(self, x1, x2):
        x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4).squeeze(2)

    def split_and_reshape(self, tensor):
        tensor1 = tensor[:, :int(self.channels * self.levels), :, :, :]
        tensor2 = tensor[:, -int(self.surface_channels):, :, :, :]
        tensor1 = tensor1.view(tensor1.shape[0], self.channels, self.levels, tensor1.shape[2], tensor1.shape[3], tensor1.shape[4])
        return tensor1, tensor2

    @classmethod
    def load_model(cls, conf):
        save_loc = conf['save_loc']
        ckpt = f"{save_loc}/checkpoint.pt"

        if not os.path.isfile(ckpt):
            raise ValueError(
                "No saved checkpoint exists. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a model with pre-trained weights from path {ckpt}"
        )

        checkpoint = torch.load(ckpt)

        if "type" in conf["model"]:
            del conf["model"]["type"]

        model_class = cls(**conf["model"])

        if conf["trainer"]["mode"] == "fsdp":
            FSDP.set_state_dict_type(
                model_class,
                StateDictType.SHARDED_STATE_DICT,
            )
            state_dict = {
                "model_state_dict": model_class.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model_class.load_state_dict(state_dict["model_state_dict"])
        else:
            model_class.load_state_dict(checkpoint["model_state_dict"])

        return model_class

    def save_model(self, conf):
        save_loc = conf['save_loc']
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))

    def rk4(self, x):

        def integrate_step(x, k, factor):
            return self.model(x + k * factor)

        k1 = self.model(x)
        k2 = integrate_step(x, k1, 0.5)
        k3 = integrate_step(x, k2, 0.5)
        k4 = integrate_step(x, k3, 1.0)

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
