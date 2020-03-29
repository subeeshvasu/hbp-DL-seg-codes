import numpy as np
import os
import torch

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