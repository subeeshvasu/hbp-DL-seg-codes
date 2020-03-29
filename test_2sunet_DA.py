import os
import torch
import numpy as np
import imageio

from torch.utils.data import DataLoader

from src.network_utils import UNetConfig
from src.networks import SharedTwoStreamUNet2Out
from src.setup import load_experiment
from src.data import loadTestDataWithoutLabelsWithFileName
from src.utils import to_save_as_uint8, load_models
from src.tester import predict_in_blocks_UNetDA2Out
from src import viz_util
from src.setup import Struct as dict_to_struct

# set device
device = "cuda"
device = "cpu"

# Load default experiment settings
experiment_path = "sandbox/experiments/test_UNet_DA.yaml"
print("Loading experiment from {}".format(experiment_path))
experiment = load_experiment(experiment_path)

# change the test settings here

test_settings = {"data_test":
                    {"dir":"./sample_test_images/", 
                    "suffix":".png"},
                 "save_dir": "./testouts/2sUNetDA",
                 "save_suffix": ".jpg",
                 "weight_dir": "./weights/2sUNetDA",
                 "weight_name": "checkpoint-pretrained.pth.tar",
                 "layer_sharing_specification": "r,r,s,s"}

test_settings = dict_to_struct(**test_settings)

# Setup net
unet_config = UNetConfig(**experiment.unet_config.__dict__)
net_seg = SharedTwoStreamUNet2Out(unet_config, layer_sharing_spec=test_settings.layer_sharing_specification).to(device)

# create a dictionary to load weights of seg net
model_dict = {}
model_dict['net_seg']    = net_seg

# Set test data
target_testset = loadTestDataWithoutLabelsWithFileName(test_settings.data_test,
                               normalize_type = "Max255")
data_loader_test = DataLoader(target_testset, batch_size=1, shuffle=False)

## set directories
img_dir = test_settings.save_dir
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

# Load net weight
checkpoint = torch.load(os.path.join(test_settings.weight_dir, test_settings.weight_name))
load_models(model_dict, checkpoint)

net_seg.eval()

for i_val, (images_val, image_name) in enumerate(data_loader_test):
    viz_util.progress_bar(i_val,len(data_loader_test),"testing progress")
    with torch.no_grad():
        images_val = images_val.data.numpy()
        images_val = np.transpose(images_val, (0, ) +tuple(range(2, 4)) + (1, ))
        images_val = images_val[0]

        prediction = predict_in_blocks_UNetDA2Out(net_seg, images_val, in_shape = [512,512], block_shape = [324, 324], 
                                        output_function=net_seg.forward_target, image_format="NHWC", device = device)
        prediction = np.transpose(prediction, (2, ) +tuple(range(0, 2)))

        prediction = np.argmax(prediction, axis = 0)
        
        predicted_labels_to_save = to_save_as_uint8(prediction)

        basename = ''.join(image_name)
        prefix, ext = os.path.splitext(basename)
        save_name = prefix + test_settings.save_suffix

        imageio.imwrite(os.path.join(img_dir,save_name),predicted_labels_to_save)