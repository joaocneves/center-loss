import torch
import numpy as np
from dataset import Dataset, create_datasets_with_attributes, create_gallery_probe_datasets, DatasetAttributes
from imageaug import transform_for_infer, transform_for_training
from utils.aux_fun_model import load_model
from utils.aux_create_sets import create_lfw_blufr_gallery_probes_sets

""" Base Functions """

def _extract_features_from_torchdataset(model, Dataset, batch_size=128, device='cuda:0'):
    """
    
        Extracts the features from a Torch Dataset object
        Example:
            
            model = load_model(arch, weights_file, device)
            gallery_set, probes_set = create_lfw_blufr_gallery_probes_sets(...)
            gallery_dataset = Dataset(gallery_set, transform_for_infer(model.IMAGE_SHAPE))
            _extract_features_from_torchdataset(model, gallery_dataset)
            
    
    """

    dataloader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=False
    )

    batch = 0
    features = []
    targets = []
    names = []

    for images, _targets, _names in dataloader:
        batch += 1
        _targets = torch.tensor(_targets).to(device)
        images = images.to(device)

        _logits, _features = model(images)

        features.append(_features.detach().cpu())
        targets.append(_targets.detach().cpu())
        names.append(_names)

    features_mat = np.vstack(features)
    targets_mat = np.hstack(targets)

    return features_mat, targets_mat, names


def _extract_features_with_attributes_from_torchdataset(model, Dataset, batch_size=128, device='cuda:0'):
    """

        Extracts the features from a Torch Dataset object
        Example:

            model = load_model(arch, weights_file, device)
            trainining_set, _ = create_datasets_with_attributes(...)
            training_dataset = Dataset(trainining_set, transform_for_infer(model.IMAGE_SHAPE))
            _extract_features_with_attributes_from_torchdataset(model, training_dataset)


    """
    
    
    training_dataloader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=batch_size,
        num_workers=6,
        shuffle=False
    )

    batch = 0
    features = []
    targets = []
    attributes = []
    names = []

    for images, _targets, _names, _image_attributes, _person_attributes in training_dataloader:
        batch += 1
        _targets = torch.tensor(_targets).to(device)
        images = images.to(device)

        _logits, _features = model(images)

        features.append(_features.detach().cpu())
        targets.append(_targets.detach().cpu())
        attributes.append(_person_attributes['mean'].detach().cpu())
        names.append(_names)

    features_mat = np.vstack(features)
    attributes_mat = np.vstack(attributes)
    targets_mat = np.hstack(targets)

    return features_mat, attributes_mat, targets_mat


""" High-Level Functions for Complete Dataset Feature Extraction"""


def extract_features(model, dataset_dir, batch_size=128, device='cpu'):
    training_set, validation_set, num_classes = create_datasets(dataset_dir, train_val_split=1)

    training_dataset = Dataset(
        training_set, transform_for_infer(model.IMAGE_SHAPE))

    features_mat, targets_mat = _extract_features_from_torchdataset(model, training_dataset, batch_size, device)

    return features_mat, targets_mat


def extract_features_with_attributes(model, dataset_name, dataset_dir, attributes_file, batch_size=128, device='cpu'):
    training_set, validation_set, num_classes = \
        create_datasets_with_attributes(dataset_name, dataset_dir, attributes_file, train_val_split=1)

    training_dataset = DatasetAttributes(training_set, transform_for_infer(model.IMAGE_SHAPE))

    features_mat, attributes_mat, targets_mat = \
        _extract_features_with_attributes_from_torchdataset(model, training_dataset, batch_size, device)

    return features_mat, attributes_mat, targets_mat

# Wrapper
def extract_features(dataset_name, dataset_dir, arch, weights_file, attributes_file=None, batch_size=128, device='cpu'):
    model = load_model(arch, weights_file, device)
    if attributes_file is None:
        features_mat, attributes_mat, targets_mat = extract_features(model, dataset_name, dataset_dir, batch_size,
                                                                     device)
        return features_mat, targets_mat
    else:
        features_mat, attributes_mat, targets_mat = extract_features_with_attributes(model, dataset_name, dataset_dir,
                                                                                     attributes_file, batch_size,
                                                                                     device)
        return features_mat, attributes_mat, targets_mat



""" High-Level Functions for Gallery/Probes Dataset Feature Extraction"""


def extract_features_identification(model, dataset_dir, batch_size=128, device='cpu'):
    gallery_set, probe_set = create_gallery_probe_datasets(dataset_dir)

    gallery_dataset = Dataset(
        gallery_set, transform_for_infer(model.IMAGE_SHAPE))

    probe_dataset = Dataset(
        probe_set, transform_for_infer(model.IMAGE_SHAPE))

    gallery_features_mat, gallery_targets_mat, _ = \
        _extract_features_from_torchdataset(model, gallery_dataset, batch_size=batch_size, device=device)
    probe_features_mat, probe_targets_mat, _ = \
        _extract_features_from_torchdataset(model, probe_dataset, batch_size=batch_size, device=device)

    return gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat



def extract_features_lfw_blufr_identification(model, dataset_dir, blufr_lfw_config_file, batch_size=128, device='cpu'):

    gallery_set, probe_set = create_lfw_blufr_gallery_probes_sets(dataset_dir, blufr_lfw_config_file, closed_set=False)

    gallery_dataset = Dataset(
        gallery_set, transform_for_infer(model.IMAGE_SHAPE))

    probe_dataset = Dataset(
        probe_set, transform_for_infer(model.IMAGE_SHAPE))

    gallery_features_mat, gallery_targets_mat, _ = \
        _extract_features_from_torchdataset(model, gallery_dataset, batch_size=batch_size, device=device)
    probe_features_mat, probe_targets_mat, _ = \
        _extract_features_from_torchdataset(model, probe_dataset, batch_size=batch_size, device=device)

    return gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat


def extract_features_identification(dataset_dir, arch, weights_file, attributes_file=None, batch_size=128,
                                    device='cpu'):
    model = load_model(arch, weights_file, device)
    # if attributes_file is None:
    gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat = extract_features_identification(
        model=model, dataset_dir=dataset_dir, batch_size=batch_size, device=device)

    return gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat









