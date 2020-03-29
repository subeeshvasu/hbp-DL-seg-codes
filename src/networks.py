from itertools import chain
import torch
from .network_utils import *

"""
Variants of UNet
subeesh.vasu@epfl.ch
CVLab EPFL 2020
"""
class UNet(nn.Module):
    """Vanilla UNet"""

    def __init__(self, unet_config=None):

        super(UNet, self).__init__()

        if unet_config is None:
            unet_config = UNetConfig()

        self.config = unet_config
        ndims = self.config.ndims

        if ndims == 2:
            self.max_pool = nn.MaxPool2d(2)
            ConvLayer = nn.Conv2d
            ConvTransposeLayer = nn.ConvTranspose2d
        elif ndims == 3:
            self.max_pool = nn.MaxPool3d(2)
            ConvLayer = nn.Conv3d
            ConvTransposeLayer = nn.ConvTranspose3d
        else:
            raise "Unet only works in 2 or 3 dimensions"

        first_layer_channels = self.config.first_layer_channels
        two_sublayers = self.config.two_sublayers
        layer1 = UNetLayer(self.config.num_input_channels,
                           first_layer_channels,
                           ndims=ndims,
                           two_sublayers=two_sublayers,
                           border_mode=self.config.border_mode)

        # Down layers.
        down_layers = [layer1]
        for i in range(1, self.config.steps + 1):
            lyr = UNetLayer(first_layer_channels * 2**(i - 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)

            down_layers.append(lyr)

        # Up layers
        up_layers = []
        for i in range(self.config.steps - 1, -1, -1):
            # Up-convolution
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1),
                                        out_channels=first_layer_channels * 2**i,
                                        kernel_size=2,
                                        stride=2)
            lyr = UNetLayer(first_layer_channels * 2**(i + 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)

            up_layers.append((upconv, lyr))

        final_layer = ConvLayer(in_channels=first_layer_channels,
                                out_channels=self.config.num_classes,
                                kernel_size=1)

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.final_layer = final_layer

    def forward(self, input, return_feature_maps=False):

        x = self.down_layers[0](input)
        
        feature_maps = []
        
        down_outputs = [x]
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)
        
        feature_maps.extend(down_outputs)
        
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            
            if not self.config.remove_skip_connections:
                x = crop_and_merge(down_output, x)
            else:
                aux = torch.zeros_like(down_output)
                x = crop_and_merge(aux, x)
            x = unet_layer(x)
            feature_maps.append(x)

        x = self.final_layer(x)
        feature_maps.append(x)
        
        if not return_feature_maps:
            return x
        else:
            return x, feature_maps

class UNetSharedSeg2Out(nn.Module):
    """Shared UNet with Two Outputs."""
    
    def __init__(self, unet_config=None, num_classes = None):
        
        super(UNetSharedSeg2Out, self).__init__()
        
        if unet_config is None:
            unet_config = UNetConfig()
        
        self.config = unet_config
        ndims = self.config.ndims
        if num_classes == None:
            self.class_nb = self.config.num_classes
        else:
            self.class_nb = num_classes

        
        if ndims == 2:
            self.max_pool = nn.MaxPool2d(2)
            ConvLayer = nn.Conv2d
            ConvTransposeLayer = nn.ConvTranspose2d
        elif ndims == 3:
            self.max_pool = nn.MaxPool3d(2)
            ConvLayer = nn.Conv3d
            ConvTransposeLayer = nn.ConvTranspose3d
        else:
            raise "Unet only works in 2 or 3 dimensions"
        
        first_layer_channels = self.config.first_layer_channels
        two_sublayers = self.config.two_sublayers
        layer1 = UNetLayer(self.config.num_input_channels,
                           first_layer_channels,
                           ndims=ndims,
                           two_sublayers=two_sublayers,
                           border_mode=self.config.border_mode)
        
        # Down layers.
        down_layers = [layer1]
        for i in range(1, self.config.steps + 1):

            lyr = UNetLayer(first_layer_channels * 2**(i - 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)
            
            down_layers.append(lyr)

        
        # Up layers
        up_layers = []
        for i in range(self.config.steps - 1, -1, -1):
            # Up-convolution
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1),
                                        out_channels=first_layer_channels * 2**i,
                                        kernel_size=2,
                                        stride=2)
            lyr = UNetLayer(first_layer_channels * 2**(i + 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)
            
            up_layers.append((upconv, lyr))
        
        classfcn_layer2 = ConvLayer(in_channels=first_layer_channels * 2**(0+1),
                                out_channels=self.class_nb,
                                kernel_size=1)
        classfcn_layer1 = ConvLayer(in_channels=first_layer_channels,
                                out_channels=self.class_nb,
                                kernel_size=1)
        
        self.down_layers = down_layers
        self.up_layers = up_layers
        
        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.classfcn_layer2 = classfcn_layer2
        self.classfcn_layer1 = classfcn_layer1
        self.softmax = nn.Softmax2d()
    
    def forward(self, input):

        low = self.down_layers[0](input)
        
        down_outputs = [low]
        x = low
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)
 
        
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers[:-1], down_outputs[-2:0:-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)
        x2 = self.classfcn_layer2(x)
        
        (upconv_layer, unet_layer) = self.up_layers[-1]
        down_output = down_outputs[0]

        x = upconv_layer(x)
        x = crop_and_merge(down_output, x)
        rec = unet_layer(x)

        x1 = self.classfcn_layer1(rec)
        return low, rec, x1, x2

class UNetSharedSeg2OutFM(nn.Module):
    """Shared UNet that returns Two Outputs and feature maps."""

    def __init__(self, unet_config=None, num_classes = None):

        super(UNetSharedSeg2OutFM, self).__init__()

        if unet_config is None:
            unet_config = UNetConfig()

        self.config = unet_config
        ndims = self.config.ndims

        if num_classes == None:
            self.class_nb = self.config.num_classes
        else:
            self.class_nb = num_classes

        if ndims == 2:
            self.max_pool = nn.MaxPool2d(2)
            ConvLayer = nn.Conv2d
            ConvTransposeLayer = nn.ConvTranspose2d
        elif ndims == 3:
            self.max_pool = nn.MaxPool3d(2)
            ConvLayer = nn.Conv3d
            ConvTransposeLayer = nn.ConvTranspose3d
        else:
            raise "Unet only works in 2 or 3 dimensions"

        first_layer_channels = self.config.first_layer_channels
        two_sublayers = self.config.two_sublayers
        layer1 = UNetLayer(self.config.num_input_channels,
                           first_layer_channels,
                           ndims=ndims,
                           two_sublayers=two_sublayers,
                           border_mode=self.config.border_mode)

        # Down layers.
        down_layers = [layer1]
        for i in range(1, self.config.steps + 1):
            lyr = UNetLayer(first_layer_channels * 2**(i - 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)

            down_layers.append(lyr)

        # Up layers
        up_layers = []
        for i in range(self.config.steps - 1, -1, -1):
            # Up-convolution
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1),
                                        out_channels=first_layer_channels * 2**i,
                                        kernel_size=2,
                                        stride=2)
            lyr = UNetLayer(first_layer_channels * 2**(i + 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)

            up_layers.append((upconv, lyr))

        classfcn_layer2 = ConvLayer(in_channels=first_layer_channels * 2**(0+1),
                                out_channels=self.class_nb,
                                kernel_size=1)
        final_layer = ConvLayer(in_channels=first_layer_channels,
                                out_channels=self.class_nb,
                                kernel_size=1)

        self.down_layers = down_layers
        self.up_layers = up_layers

        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.final_layer = final_layer
        self.classfcn_layer2 = classfcn_layer2

    def forward(self, input, return_feature_maps=False):

        low = self.down_layers[0](input)
        x = low
        
        feature_maps = []
        
        down_outputs = [x]
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)
        
        feature_maps.extend(down_outputs)
        
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers[:-1], down_outputs[-2:0:-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)
            feature_maps.append(x)

        x2 = self.classfcn_layer2(x)
        
        (upconv_layer, unet_layer) = self.up_layers[-1]
        down_output = down_outputs[0]

        x = upconv_layer(x)
        x = crop_and_merge(down_output, x)
        rec = unet_layer(x)
        feature_maps.append(rec)

        x1 = self.final_layer(rec)
        feature_maps.append(x1)
        
        return low, rec, x1, x2, feature_maps

"""
Two-Stream UNet
roger.bermudez@epfl.ch
CVLab EPFL 2019
"""
class SharedTwoStreamUNet2Out(torch.nn.Module):
    """Two Stream UNet with two outputs."""
    def __init__(self, unet_config=None, layer_sharing_spec=None):
        super().__init__()
        self.config = unet_config

        self.unet_source = UNetSharedSeg2OutFM(unet_config)
        self.unet_target = UNetSharedSeg2OutFM(unet_config)

        if not layer_sharing_spec:
            layer_sharing_spec = '-' * TwoStreamUNet._get_num_layers(self.unet_source)
        self.layers_to_regularize, self.layers_to_share = TwoStreamUNet.parse_layer_sharing(self, layer_sharing_spec)
        self.feature_maps_to_regularize = [False] * (len(self.layers_to_regularize) - 1) + [True]

        self.features_source = None
        self.features_target = None

        self.regularization_params = self._get_regularization_params()
        for parameter_pair in self.regularization_params:
            if parameter_pair is not None:
                (weight_name, weight_param), (bias_name, bias_param) = parameter_pair
                self.register_parameter(weight_name, weight_param)
                self.register_parameter(bias_name, bias_param)
        self.share_weights()  # To be called after loading pretrained model

    def _get_regularization_params(self):
        """
        Returns a list of pairs (weights, biases), one for each layer, for weight regularization.
        """
        all_source_layers = TwoStreamUNet._get_all_layers(self.unet_source)
        all_shared_params = (((f'reg_layer_{layer_num}_w', torch.nn.Parameter(torch.ones(1)).to(self.device)),
                              (f'reg_layer_{layer_num}_b', torch.nn.Parameter(torch.zeros(1)).to(self.device)))
                             if apply_regularization else None
                             for ((layer_num, layer), apply_regularization) in
                             zip(enumerate(all_source_layers), self.layers_to_regularize))

        return list(all_shared_params)

    def share_weights(self):
        """ Copies references to shared weights, according to specification from layers_to_share """
        param_pairs = zip(*map(TwoStreamUNet._get_all_layers, (self.unet_source, self.unet_target)))
        shared_params = (param_pair for param_pair, shared in zip(param_pairs, self.layers_to_share) if shared)
        for layer_source, layer_target in shared_params:
            if isinstance(layer_source, UNetLayer):
                layer_source = layer_source.unet_layer
                layer_target = layer_target.unet_layer
            if hasattr(layer_source, "weight"):
                layer_target.weight = layer_source.weight
                layer_target.bias = layer_source.bias
            for operation_source, operation_target in zip(layer_source.children(), layer_target.children()):
                if hasattr(operation_source, "weight"):
                    operation_target.weight = operation_source.weight
                    operation_target.bias = operation_source.bias

    def forward(self, batch_source, batch_target):
        low_s, rec_s, x1_s, x2_s, self.features_source = self.unet_source.forward(batch_source)
        low_t, rec_t, x1_t, x2_t, self.features_target = self.unet_target.forward(batch_target)
        return low_s, rec_s, x1_s, x2_s, low_t, rec_t, x1_t, x2_t

    def forward_source(self, batch_source):
        low_s, rec_s, x1_s, x2_s, self.features_source = self.unet_source.forward(batch_source)
        return low_s, rec_s, x1_s, x2_s

    def forward_target(self, batch_target):
        low_t, rec_t, x1_t, x2_t, self.features_target = self.unet_target.forward(batch_target)
        return low_t, rec_t, x1_t, x2_t

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        # Restore parameter sharing!
        self.share_weights()

        return self

    @property
    def device(self):
        return next(self.parameters()).device