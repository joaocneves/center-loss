
import torch
import numpy as np
from dataset import Dataset, create_datasets_with_attributes, create_gallery_probe_datasets, DatasetAttributes
from imageaug import transform_for_infer, transform_for_training
from models import Resnet50FaceModel, Resnet18FaceModel

def load_model(arch, weights_file, device):
    if arch == 'resnet18':
        model_class = Resnet18FaceModel
    if arch == 'resnet50':
        model_class = Resnet50FaceModel

    checkpoint = torch.load(weights_file)
    num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]

    model = model_class(num_classes, feat_normalization='none').to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    return model

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

def _extract_features_from_torchdataset(model, Dataset, batch_size=128, device='cuda:0'):

    dataloader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=batch_size,
        num_workers=0,
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

    return features, targets, names



def extract_features(model, dataset_dir, batch_size=128, device='cpu'):

    training_set, validation_set, num_classes = create_datasets(dataset_dir, train_val_split=1)

    training_dataset = Dataset(
        training_set, transform_for_infer(model.IMAGE_SHAPE))

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    batch = 0
    features = []
    targets = []
    names = []

    for images, _targets, _names, _image_attributes, _person_attributes in training_dataloader:
        batch += 1
        _targets = torch.tensor(_targets).to(device)
        images = images.to(device)

        _logits, _features = model(images)

        features.append(_features.detach().cpu())
        targets.append(_targets.detach().cpu())
        names.append(_names)

    features_mat = np.vstack(features)
    targets_mat = np.hstack(targets)

    return features_mat, targets_mat


def _extract_features_identification(model, dataset_dir, batch_size=128, device='cpu'):

    gallery_set, probe_set = create_gallery_probe_datasets(dataset_dir)

    gallery_dataset = Dataset(
        gallery_set, transform_for_infer(model.IMAGE_SHAPE))

    probe_dataset = Dataset(
        probe_set, transform_for_infer(model.IMAGE_SHAPE))

    gfeatures, gtargets, _ = _extract_features_from_torchdataset(model, gallery_dataset, batch_size=batch_size, device=device)
    pfeatures, ptargets, _ = _extract_features_from_torchdataset(model, probe_dataset, batch_size=batch_size, device=device)

    gallery_features_mat = np.vstack(gfeatures)
    gallery_targets_mat = np.hstack(gtargets)

    probe_features_mat = np.vstack(pfeatures)
    probe_targets_mat = np.hstack(ptargets)

    return gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat

def extract_features_with_attributes(model, dataset_name, dataset_dir, attributes_file, batch_size=128, device='cpu'):

    training_set, validation_set, num_classes = create_datasets_with_attributes(dataset_name, dataset_dir, attributes_file,
                                                                                    train_val_split=1)
    training_dataset = DatasetAttributes(
        training_set, transform_for_infer(model.IMAGE_SHAPE))

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
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


def extract_features_identification(dataset_name, dataset_dir, arch, weights_file, attributes_file=None, batch_size=128, device='cpu'):

    model = load_model(arch, weights_file, device)
    #if attributes_file is None:
    gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat = _extract_features_identification(model=model, dataset_dir=dataset_dir, batch_size=batch_size, device=device)

    return gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat


def extract_features(dataset_name, dataset_dir, arch, weights_file, attributes_file=None, batch_size=128, device='cpu'):

    model = load_model(arch, weights_file, device)
    if attributes_file is None:
        features_mat, attributes_mat, targets_mat = extract_features(model, dataset_name, dataset_dir, batch_size, device)
        return features_mat, targets_mat
    else:
        features_mat, attributes_mat, targets_mat = extract_features_with_attributes(model, dataset_name, dataset_dir, attributes_file, batch_size, device)
        return features_mat, attributes_mat, targets_mat

def extract_features_train_val(dataset_dir, arch, weights_file, device):

    model = load_model(arch, weights_file, device)
    training_set, validation_set, num_classes = create_datasets(dataset_dir)

    training_dataset = Dataset(
        training_set, transform_for_training(model.IMAGE_SHAPE))
    validation_dataset = Dataset(
        validation_set, transform_for_infer(model.IMAGE_SHAPE))

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=128,
        num_workers=6,
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=128,
        num_workers=6,
        shuffle=False
    )

    batch = 0
    features = []
    targets = []

    for images, _targets, names in validation_dataloader:
        batch += 1
        _targets = torch.tensor(_targets).to(device)
        images = images.to(device)

        _logits, _features = model(images)

        features.append(_features.detach().cpu())
        targets.append(_targets.detach().cpu())

    features_mat = np.vstack(features)
    targets_mat = np.hstack(targets)

    return features_mat, targets_mat

def load_attributes(atributes_file):
    soft_feat = np.load(atributes_file)
    soft_feat = soft_feat.astype('float32')

    return soft_feat