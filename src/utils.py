import numpy as np
import os
import torch
import torch.nn.functional as F

from src.network_utils import UNetConfig
from src.networks import SharedTwoStreamUNet2Out, UNetSharedSeg2Out


def to_save_as_uint8(image):
    image = image - image.min()
    if not(image.max() == 0):
        image = image / image.max()
    image = 255 * image
    image = image.astype(np.uint8)
    return image

def load_models(model_dict, loaded_checkpoint):
    pretrained_dict = loaded_checkpoint
    for key, value in model_dict.items():
        value.load_state_dict(pretrained_dict[key])
        print('loading model - {}'.format(key))

def save_checkpoint_with_dict_simple(model_dict, checkpoint_name, save_dir):
    state = {}

    update_model_dict = {key: value.state_dict() for key, value in model_dict.items()}
    state.update(update_model_dict)

    filename = os.path.join(
        save_dir, "checkpoint-{}.pth.tar".format(checkpoint_name)
    )
    torch.save(state, filename)
    print('saving checkpoint - {}'.format(checkpoint_name))


class UNetSharedSeg2OutPredictionOnly(UNetSharedSeg2Out):
    def forward(self, input):
        low, rec, x1, x2 = super().forward(input)
        return F.softmax(x1, dim=1)

def get_unet(**kwargs):
    unet_config = UNetConfig(**kwargs)
    net = UNetSharedSeg2OutPredictionOnly(unet_config)
    return net


class SharedTwoStreamUNet2OutPredictionOnly(SharedTwoStreamUNet2Out):
    def forward(self, batch_target):
        low_t, rec_t, x1_t, x2_t = super().forward_target(batch_target)
        return F.softmax(x1_t, dim=1)


def get_2sunet(layer_sharing_specification: str, **kwargs):
    unet_config = UNetConfig(**kwargs)
    net_seg = SharedTwoStreamUNet2OutPredictionOnly(unet_config, layer_sharing_spec=layer_sharing_specification)
    return net_seg
