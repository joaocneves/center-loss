import os
from os.path import exists

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.manifold import TSNE

from utils.aux_fun_inference import extract_features_train_val, load_attributes, extract_features
from utils.aux_func_distances import fast_pairwise_dist

## Script for visualizing each image in the feature space (2D manifold) FOR VISUALISING EACH IMAGE IN THE FEATURE SPACE (2D MANIFOLD)
## Each sample is assigned a color with respect to its attributes

DATASET_DIR = '../datasets/celeba/celeba-aligned_insight/'
#DATASET_DIR = '/home/socialab/Desktop/Joao/datasets/lfw-aligned_insight'
arch = 'resnet50'
weights_file = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs/center-loss.imgsize_112/epoch_30.pth.tar'
#weights_file = '../logs/attribute-loss.imgsize_112_attweight_0.1/epoch_18.pth.tar'
#weights_file = '../logs/center-loss.imgsize_112_centerloss/epoch_18.pth.tar'
device = 'cuda:0'
#atributes_file = '../datasets/lfw/atts_lfw.npy'
atributes_file = '../datasets/celeba/atts_celeba.npy'


os.makedirs('tmp_data', exist_ok=True)

if not exists('tmp_data/celeba_feats_klnet.npy'):
    features_mat, targets_mat = extract_features(DATASET_DIR, arch, weights_file, device)
    np.save('tmp_data/celeba_feats_klnet.npy', features_mat)
    np.save('tmp_data/celeba_targets_klnet.npy', targets_mat)
else:
    targets_mat = np.load('tmp_data/celeba_targets_klnet.npy')
    features_mat = np.load('tmp_data/celeba_feats_klnet.npy')

attributes_mat = load_attributes(atributes_file)

#features_mat, targets_mat = extract_features_train_val(DATASET_DIR, arch, weights_file, device)
#attributes_mat = load_attributes(atributes_file)

# attributes_mat[:,9] -> blond hair
# attributes_mat[:,15] -> eyeglasses
# attributes_mat[:,20] -> male

features_mat = features_mat[5000:6400,:]
targets_mat = targets_mat[5000:6400]
attributes_mat = attributes_mat[targets_mat,:]
features_dist = fast_pairwise_dist(features_mat, 'cuda:0')

hair = attributes_mat[:,0]
eye = attributes_mat[:,1]
male = attributes_mat[:,2]

att = attributes_mat[:,[20,25,15]]
att = np.around(att,decimals=0)
att_dist = fast_pairwise_dist(att, 'cpu')
mds = manifold.MDS(2, dissimilarity='precomputed')
coords = mds.fit_transform(att_dist)


#tsne = TSNE(n_components=2, random_state=123)
#coords = tsne.fit_transform(features_mat)

x, y = coords[:, 0], coords[:, 1]

fig, ax = plt.subplots()
for i in np.unique(att, axis=0):

    ax.scatter(x[(att==i).all(axis=1)], y[(att==i).all(axis=1)])
    ax.text(x[(att==i).all(axis=1)][0], y[(att==i).all(axis=1)][0], str(i), style ='italic', fontsize = 9, color ="green")
#ax.scatter(x, y)
#for (city, _x, _y) in zip(targets_mat, x, y):
#    ax.annotate(city, (_x, _y))
plt.show()