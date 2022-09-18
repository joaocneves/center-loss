import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform

from aux_fun_inference import extract_features_train_val, load_attributes, extract_features





#DATASET_DIR = '../datasets/celeba/celeba-aligned_insight/'
DATASET_DIR = '/home/socialab/Desktop/Joao/datasets/lfw-aligned_insight'
arch = 'resnet50'
weights_file = '../logs/center-loss.imgsize_112_centerloss/epoch_18.pth.tar'
#weights_file = '../logs/attribute-loss.imgsize_112_attweight_0.2/epoch_18.pth.tar'
device = 'cuda:0'
atributes_file = '../datasets/celeba/atts_celeba.npy'

features_mat, targets_mat = extract_features(DATASET_DIR, arch, weights_file, device)
attributes_mat = load_attributes(atributes_file)
attributes_mat = attributes_mat[targets_mat,[20]]
attributes_mat = np.expand_dims(attributes_mat, axis=1)
attributes_mat = np.round(attributes_mat)



ranks, att_dist_rank = attribute_consistency(features_mat, attributes_mat)

plt.plot(ranks[1:10], att_dist_rank[1:10])
plt.show()