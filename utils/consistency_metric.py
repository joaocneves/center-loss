import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

import torch
from dataset import Dataset, create_datasets
from imageaug import transform_for_infer, transform_for_training
from models import Resnet50FaceModel, Resnet18FaceModel

def consistency_metric(embeddings, attributes):
    emb_dist = squareform(pdist(embeddings))
    att_dist = squareform(pdist(attributes))

    thresholds = np.unique(emb_dist)
    avg_dist_global = []

    for th in thresholds:
        neighbours_th = emb_dist <= th

        att_dist_th = np.where(neighbours_th == True, att_dist, np.nan)

        sum = np.nansum(att_dist_th, axis=1)
        count = np.sum(neighbours_th, axis=1)

        avg_dist_per_sample = sum/count

        avg_dist_global.append(np.average(avg_dist_per_sample))

    avg_dist_global = np.array(avg_dist_global)

    return thresholds, avg_dist_global


def load_model(arch, weights_file):

    if arch == 'resnet18':
        model_class = Resnet18FaceModel
    if arch == 'resnet50':
        model_class = Resnet50FaceModel

    checkpoint = torch.load(weights_file)
    num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]

    model = model_class(num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    return model


DATASET_DIR = '../datasets/celeba/celeba-aligned_insight/'
arch = 'resnet50'
weights_file = '../logs/center-loss.img_size_112/epoch_14.pth.tar'
device = 'cuda:0'

model = load_model(arch, weights_file)

training_set, validation_set, num_classes = create_datasets(DATASET_DIR)

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

for images, targets, names in training_dataloader:
    batch += 1
    targets = torch.tensor(targets).to(device)
    images = images.to(device)

    logits, features = model(images)







embeddings = np.random.randint(5, size=(120, 64))
attributes = np.random.randint(2, size=(120, 3))

thresholds, avg_dist_global = consistency_metric(embeddings, attributes)

plt.plot(thresholds, avg_dist_global)
plt.show()
