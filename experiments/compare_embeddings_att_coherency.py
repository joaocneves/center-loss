"""
This scripts evaluate the embeddings coherency with respect to the attributes of each image
Two strategies cna be used: Rank or Distance
"""
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from dataset import Dataset, create_datasets
from imageaug import transform_for_infer, transform_for_training
from models import Resnet50FaceModel, Resnet18FaceModel
from utils.aux_fun_inference import extract_features, load_attributes
from utils.metrics.consistency_metric import consistency_metric_distance

DATASET_DIR = '../../datasets/celeba/celeba-aligned_insight/'
arch = 'resnet50'
weights_file = '../logs/center-loss.img_size_112/epoch_14.pth.tar'
device = 'cuda:0'
atributes_file = '../../datasets/celeba/atts_celeba.npy'


features_mat, targets_mat = extract_features(DATASET_DIR, arch, weights_file, device)
attributes_mat = load_attributes(atributes_file, device)
attributes_mat = attributes_mat[targets_mat,:]

#features_mat = torch.tensor(features_mat).to(device)
#attributes_mat = torch.tensor(attributes_mat).to(device)

thresholds, avg_dist_global = consistency_metric_distance(features_mat, attributes_mat)
data = dict()
data['th'] = thresholds
data['avg'] = avg_dist_global
torch.save(data, 'epoch14.pt')
plt.plot(thresholds, avg_dist_global)

#
#
DATASET_DIR = '../../datasets/celeba/celeba-aligned_insight/'
arch = 'resnet50'
weights_file = '../logs/attribute-loss.img_size_112/epoch_14.pth.tar'
device = 'cuda:0'
atributes_file = '../../datasets/celeba/atts_celeba.npy'

features_mat, targets_mat = extract_features(DATASET_DIR, arch, weights_file, device)
attributes_mat = load_attributes(atributes_file, device)
attributes_mat = attributes_mat[targets_mat,:]

features_mat = torch.tensor(features_mat).to(device)
attributes_mat = torch.tensor(attributes_mat).to(device)

thresholds, avg_dist_global = consistency_metric_distance(features_mat, attributes_mat)
data = dict()
data['th'] = thresholds
data['avg'] = avg_dist_global
torch.save(data, 'epoch66.pt')

plt.plot(thresholds, avg_dist_global)
plt.show()