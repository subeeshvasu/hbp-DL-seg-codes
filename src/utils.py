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
    return UNetSharedSeg2OutPredictionOnly(unet_config)


class SharedTwoStreamUNet2OutPredictionOnly(SharedTwoStreamUNet2Out):
    def __init__(self, *, input_is_from_source_domain, **super_kwargs):
        super().__init__(**super_kwargs)
        if input_is_from_source_domain:
            self.forward = self.forward_source
        else:
            self.forward = self.forward_target

    def forward_source(self, batch_target):
        low_s, rec_s, x1_s, x2_s = super().forward_source(batch_target)
        return F.softmax(x1_s, dim=1)

    def forward_target(self, batch_target):
        low_t, rec_t, x1_t, x2_t = super().forward_target(batch_target)
        return F.softmax(x1_t, dim=1)


def get_2sunet(layer_sharing_specification: str, input_is_from_source_domain: bool, **kwargs):
    unet_config = UNetConfig(**kwargs)
    return SharedTwoStreamUNet2OutPredictionOnly(
        unet_config=unet_config,
        layer_sharing_spec=layer_sharing_specification,
        input_is_from_source_domain=input_is_from_source_domain
    )
