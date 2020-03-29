import torch
import numpy as np
from itertools import product
from torch.autograd import Variable 
import torch.nn.functional as F

def _check_and_prepare_image(image, ndims, image_format):
    
    if image_format not in ['NCHW', 'NHWC']:
        raise ValueError("image_format not in ['NCHW', 'NHWC']")
    
    if image.ndim < ndims + 1:
        # Add channel
        if image_format == 'NCHW':
            image = image[None]
        else:
            image = image[..., None]
    
    if image.ndim != ndims + 1:
        raise ValueError("invalid dimensions for the image; it should be {} or {}".format(ndims, ndims+1))
    
    return image

def centers(full, out_shape):
    
    start = out_shape // 2
    stop = full - (out_shape - out_shape // 2)
    
    count = ceil_div(stop - start, out_shape) + 1
    
    return np.int_(np.round(np.linspace(start, stop, count)))

def ceil_div(a, b):
    
    return -(-a // b)

def _box_in_bounds(box, image_shape):

    newbox = []
    pad_width = []

    for box_i, shape_i in zip(box, image_shape):

        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

        newbox.append(newbox_i)
        pad_width.append(pad_width_i)

    needs_padding = any(i != (0, 0) for i in pad_width)

    return newbox, pad_width, needs_padding

def get_patch(image, patch_shape, center, mode='constant'):

    if mode == 'reflect':
        # We need to deal with even patch shapes when the mode is reflect
        correction_slice = tuple(slice(None, None if sh & 1 else -1) for sh in patch_shape)
        patch_shape = tuple(sh | 1 for sh in patch_shape)

    box = [(i-ps//2, i-ps//2+ps) for i, ps in zip(center, patch_shape)]

    box, pad_width, needs_padding = _box_in_bounds(box, image.shape)
    slices = tuple(slice(i[0], i[1]) for i in box)

    patch = image[slices]

    if needs_padding:
        if len(pad_width) < patch.ndim:
            pad_width.append((0, 0))
        patch = np.pad(patch, pad_width, mode=mode)

    if mode == 'reflect':
        patch = patch[correction_slice]

    return patch

def predict_in_blocks_UNetDA2Out(net, image, in_shape, block_shape,
                      output_function=None,
                      image_format='NCHW',
                      pad_mode="reflect",
                      verbose=True,
                      device = "cpu"):
    """
    Returns the prediction of the U-Net for the given `image`. Processes the
    image in overlapping blocks of the given size. If the image is not very
    large, `predict_at_once` might be more convenient and faster. Use this
    function when `predict_at_once` gives memory errors.
    """
    
    ndims = net.config.ndims
    image = _check_and_prepare_image(image, ndims, image_format)
    
    if output_function is None:
        output_function = net.forward
    
    if image_format == 'NCHW':
        # Move channels to the end
        image = np.transpose(image, tuple(range(1, ndims+1)) + (0, ))
    
    # From this point, image has format NHWC
    image_shape = image.shape[:ndims] # Remove the channels in the shape
    
    if any(i < j for i, j in zip(image_shape, block_shape)):
        raise ValueError("image_shape {} is smaller than the block_shape {}; try a smaller value of hint_block_shape".format(image_shape, block_shape))
    
    grid = map(centers, image_shape, block_shape)
    
    num_channels = net.config.num_classes
    # The result also has format NHWC
    result = np.zeros(image_shape + (num_channels,), dtype=np.float32)
    
    
    for c in product(*grid):
        patch_x = get_patch(image, in_shape, c, mode=pad_mode)
        
        patch_x = np.transpose(patch_x, (ndims, ) + tuple(range(0, ndims)))
        _, _, patch_o, _ = output_function(Variable(torch.from_numpy(patch_x[None, ...])).to(device))
        patch_o = patch_o[0]
        patch_o = F.softmax(patch_o, dim=0)
        # Transpose the output so that the channels come at the end.
        patch_o = np.transpose(patch_o.cpu().data.numpy(), tuple(range(1, ndims + 1)) + (0,))
        p_center = (int(patch_o.shape[0]/ 2), int(patch_o.shape[1]/ 2))
        patch_o_c = get_patch(patch_o, block_shape, p_center)
        get_patch(result, block_shape, c)[:] = patch_o_c
        
    
    # Change format of the result if required
    if image_format == 'NCHW':
        result = np.transpose(result, (ndims, ) + tuple(range(0, ndims)))
    
    return result