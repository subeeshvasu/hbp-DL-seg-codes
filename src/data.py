import imageio
import torch.utils.data as data
import glob
import numpy as np
import os

image_file_extensions = [".png", ".tif", ".bmp", ".jpg", ".jpeg", ".tga"]
def sort_nicely(l):
        
    l_filtered = [k for k in l if os.path.splitext(k)[1].lower() in image_file_extensions]
    """ Sort the given list in the way that humans expect."""
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l_filtered, key=alphanum_key)

def labels_to_probabilities(labels, num_classes):
    probs = np.zeros((1, num_classes) + labels.shape[1:], dtype=np.float32)
    probs[(0, labels) + np.ix_(*map(range, labels.shape[1:]))] = 1
    probs = probs[0,:,:,:]
    return probs

def normalize_intensity(image, mean_bgr = 128, std_bgr = 128, normalize_type = "Max255"):
    if normalize_type == "Std":
        image = (image - mean_bgr) / std_bgr
    elif normalize_type == "Max255":
        image = (image / 255.0) * 2 - 1
    return image

def _map_labels(labels, num_classes):
    '''
    Converts labels from numerical values into classes
    '''
    new_labels = np.zeros_like(labels)
    unique_labels = np.unique(labels)
    if num_classes == 2:
        if all(label in (0, 255) for label in unique_labels):
            mappings = {0: 0, 255: 1}
        elif all(label in (0, 1) for label in unique_labels):
            mappings = {0: 0, 1: 1}
        else:
            mappings = {0: 0, 1: 1}
            labels = labels > 128

        for value, new_value in mappings.items():
            label_mask = labels == value
            new_labels[label_mask] = new_value

    else:
        new_labels = labels
        if len(unique_labels) > num_classes:
            print("Weird labels: Number of unique values {} in labels are more than the number of classes - {}".format(len(unique_labels),num_classes))

    return new_labels


def get_normalized_image(file_path, normalize_type):
    img = imageio.imread(file_path, pilmode="RGB")
    img = np.asarray(imageio.core.image_as_uint(img, bitdepth=8)).transpose(2, 0, 1)

    # normalize image intensity
    img = normalize_intensity(img, normalize_type=normalize_type).astype("float32")

    return img


def get_labels(file_path, num_classes):
    gt = imageio.imread(file_path)
    gt = np.asarray(imageio.core.image_as_uint(gt, bitdepth=8))[None, ...]

    # convert labels from numerical values into classes + compute class weights
    gt = _map_labels(gt, num_classes)

    return gt


def test_loader(file_name, images_path, labels_path, num_classes, normalize_type="Max255"):

    ### images --> img, labels --> gt

    basename = os.path.basename(file_name)
    prefix, ext = os.path.splitext(basename)
    img_basename = prefix + images_path.suffix
    gt_basename = prefix + labels_path.suffix

    img = get_normalized_image(os.path.join(images_path.dir, img_basename), normalize_type)
    gt = get_labels(os.path.join(labels_path.dir, gt_basename), num_classes)

    return img, gt


def test_loader_without_labels(file_name, images_path, normalize_type="Max255"):

    ### images --> img

    basename = os.path.basename(file_name)
    prefix, ext = os.path.splitext(basename)
    img_basename = prefix + images_path.suffix

    return get_normalized_image(os.path.join(images_path.dir, img_basename), normalize_type)


class loadTestData(data.Dataset):

    def __init__(self, images_path, labels_path, num_classes = 2, normalize_type = "Max255"): 
        test_list = sort_nicely(glob.glob(images_path.dir + "*" + images_path.suffix))
        self.file_names = test_list
        self.loader = test_loader
        self.images_path = images_path
        self.labels_path = labels_path
        self.num_classes = num_classes
        self.normalize_type = normalize_type

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img, gt = self.loader(file_name, self.images_path, self.labels_path, self.num_classes, self.normalize_type) 

        gt_prob = np.copy(gt)

        if self.num_classes > 1:
            gt_prob = labels_to_probabilities(gt_prob.astype(np.uint8), self.num_classes)
        return img, gt_prob

    def __len__(self):
        return len(self.file_names)

class loadTestDataWithoutLabels(data.Dataset):
    def __init__(self, images_path, normalize_type = "Max255"): 
        test_list = sort_nicely(glob.glob(images_path.dir + "*" + images_path.suffix))
        self.file_names = test_list
        self.loader = test_loader_without_labels
        self.images_path = images_path
        self.normalize_type = normalize_type

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = self.loader(file_name, self.images_path, self.normalize_type)
        return img

    def __len__(self):
        return len(self.file_names)

class loadTestDataWithoutLabelsWithFileName(data.Dataset):
    def __init__(self, images_path, normalize_type = "Max255"): 
        test_list = sort_nicely(glob.glob(images_path.dir + "*" + images_path.suffix))
        self.file_names = test_list
        self.loader = test_loader_without_labels
        self.images_path = images_path
        self.normalize_type = normalize_type

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = self.loader(file_name, self.images_path, self.normalize_type)
        basename = os.path.basename(file_name)
        return img, basename

    def __len__(self):
        return len(self.file_names)