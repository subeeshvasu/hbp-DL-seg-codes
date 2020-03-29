import torch
import torch.nn as nn
from torch.autograd import Variable

class UNetLayer(nn.Module):
    
    def __init__(self, num_channels_in, num_channels_out, ndims,
                 two_sublayers=True, border_mode='valid', batch_normal = True):
        
        super(UNetLayer, self).__init__()
        
        self.two_sublayers = two_sublayers
        self.batch_normal = batch_normal
        
        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        if border_mode == 'valid':
            padding = 0
        elif border_mode == 'same':
            padding = 1
        else:
            raise ValueError("unknown border_mode `{}`".format(border_mode))
        
        conv1 = conv_op(num_channels_in, num_channels_out,
                        kernel_size=3, padding=padding)
        relu1 = nn.ReLU()
        batch_norm = nn.BatchNorm2d(num_channels_out)
        
        if not self.two_sublayers:
            self.unet_layer = nn.Sequential(conv1, batch_norm, relu1)
        else:
            conv2 = conv_op(num_channels_out, num_channels_out,
                            kernel_size=3, padding=padding)
            relu2 = nn.ReLU()
            batch_norm2 = nn.BatchNorm2d(num_channels_out)


            if self.batch_normal:
                self.unet_layer = nn.Sequential(conv1, batch_norm, relu1, conv2, batch_norm2, relu2)
            else:
                self.unet_layer = nn.Sequential(conv1, relu1,conv2, relu2)

    def forward(self, x):
        
        return self.unet_layer(x)

class UNetConfig(object):

    def __init__(self,
                 steps=4,
                 first_layer_channels=64,
                 num_classes=2,
                 num_input_channels=1,
                 two_sublayers=True,
                 ndims=2,
                 border_mode='valid',
                 remove_skip_connections=False):

        if border_mode not in ['valid', 'same']:
            raise ValueError("`border_mode` not in ['valid', 'same']")

        self.steps = steps
        self.first_layer_channels = first_layer_channels
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.two_sublayers = two_sublayers
        self.ndims = ndims
        self.border_mode = border_mode
        self.remove_skip_connections = remove_skip_connections

        border = 4 if self.two_sublayers else 2
        if self.border_mode == 'same':
            border = 0
        self.first_step = lambda x: x - border
        self.rev_first_step = lambda x: x + border
        self.down_step = lambda x: (x - 1) // 2 + 1 - border
        self.rev_down_step = lambda x: (x + border) * 2
        self.up_step = lambda x: (x * 2) - border
        self.rev_up_step = lambda x: (x + border - 1) // 2 + 1
        

    def __getstate__(self):
        return [self.steps, self.first_layer_channels, self.num_classes, self.two_sublayers, self.ndims, self.border_mode]

    def __setstate__(self, state):
        return self.__init__(*state)

    def __repr__(self):
        return "{0.__class__.__name__!s}(steps={0.steps!r}, first_layer_channels={0.first_layer_channels!r}, " \
                "num_classes={0.num_classes!r}, num_input_channels={0.num_input_channels!r}, "\
                "two_sublayers={0.two_sublayers!r}, ndims={0.ndims!r}, "\
                "border_mode={0.border_mode!r})".format(self)

    def out_shape(self, in_shape):
        """
        Return the shape of the output given the shape of the input
        """

        shapes = self.feature_map_shapes(in_shape)
        return shapes[-1][1:]

    def feature_map_shapes(self, in_shape):

        def _feature_map_shapes():

            shape = np.asarray(in_shape)
            yield (self.num_input_channels,) + tuple(shape)

            shape = self.first_step(shape)
            yield (self.first_layer_channels,) + tuple(shape)

            for i in range(self.steps):
                shape = self.down_step(shape)
                channels = self.first_layer_channels * 2 ** (i + 1)
                yield (channels,) + tuple(shape)

            for i in range(self.steps):
                shape = self.up_step(shape)
                channels = self.first_layer_channels * 2 ** (self.steps - i - 1)
                yield (channels,) + tuple(shape)

            yield (self.num_classes,) + tuple(shape)

        return list(_feature_map_shapes())

    def _out_step(self):
        
        _, out_shape_0 = self.in_out_shape((0,))
        _, out_shape_1 = self.in_out_shape((out_shape_0[0] + 1,))
        return out_shape_1[0] - out_shape_0[0]

    def in_out_shape(self, out_shape_lower_bound,
                     given_upper_bound=False):
        """
        Compute the best combination of input/output shapes given the
        desired lower bound for the shape of the output
        """
        
        if given_upper_bound:
            out_shape_upper_bound = out_shape_lower_bound
            out_step = self._out_step()
            out_shape_lower_bound = tuple(i - out_step + 1 for i in out_shape_upper_bound)

        shape = np.asarray(out_shape_lower_bound)

        for i in range(self.steps):
            shape = self.rev_up_step(shape)

        # Compute correct out shape from minimum shape
        out_shape = np.copy(shape)
        for i in range(self.steps):
            out_shape = self.up_step(out_shape)

        # Best input shape
        for i in range(self.steps):
            shape = self.rev_down_step(shape)

        shape = self.rev_first_step(shape)
        
        return tuple(shape), tuple(out_shape)
    
    def margin(self):
        """
        Return the size of the margin lost around the input images as a
        consequence of the sequence of convolutions and max-poolings.
        """
        in_shape, out_shape = self.in_out_shape((0,))
        return (in_shape[0] - out_shape[0]) // 2

    def in_out_pad_widths(self, out_shape_lower_bound):

        in_shape, out_shape = self.in_out_shape(out_shape_lower_bound)

        in_pad_widths = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                            for sh_i, sh_o in zip(out_shape_lower_bound, in_shape)]
        out_pad_widths = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                            for sh_i, sh_o in zip(out_shape_lower_bound, out_shape)]

        return in_pad_widths, out_pad_widths

def crop_and_merge(tensor1, tensor2):
    
    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)
    
    return torch.cat((tensor1[slices], tensor2), 1)

def crop_slices(shape1, shape2):
    
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2)
                    for sh1, sh2 in zip(shape1, shape2)]
    return slices