import torch
import numpy as np
from dataset import Dataset
from imageaug import transform_for_infer, transform_for_training

def create_imgarray_from_set(set, image_shape):

    dataset = Dataset(set, transform_for_infer(image_shape))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=6,
        shuffle=False
    )

    batch = 0
    images = []
    targets = []

    for _images, _targets, _names in dataloader:
        batch += 1
        targets.append(_targets)
        images.append(_images)

    images = np.vstack(images)
    targets = np.hstack(targets)

    return images, targets

def create_lfw_bin_all_images(dataset_dir, image_shape):
    """This function creates a file containing all lfw images using the structure of lfw.bin of faces_emore"""

    training_set, validation_set, num_classes = create_datasets(dataset_dir, train_val_split=1)

    training_dataset = Dataset(
        training_set, transform_for_infer(image_shape))

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=128,
        num_workers=0,
        shuffle=False
    )

    batch = 0
    images = []
    targets = []

    for _images, _targets, _names in training_dataloader:
        batch += 1
        targets.append(_targets)
        images.append(_images)

    images = np.vstack(images)
    targets = np.hstack(targets)

    return images, targets