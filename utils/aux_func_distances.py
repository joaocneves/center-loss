import random

import numpy as np

import torch
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import pairwise_distances


def fast_matrix_rowcol_distance(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    N, d = x.shape
    M, d = y.shape

    x_norm = (x ** 2).sum(1)
    x_norm = np.tile(x_norm, [M, 1]).T

    if y is not None:
        y_norm = (y ** 2).sum(1)
        y_norm = np.tile(y_norm, [N, 1])
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    dist = dist ** 0.5
    return dist


def fast_pairwise_dist(mat, device):
    mat = torch.tensor(mat).to(device)
    pairwise_dist = squareform(torch.nn.functional.pdist(mat).detach().cpu().numpy())

    return pairwise_dist

def intra_inter_distances(features, labels):
    """Calculates the intra and inter class distances for a set of features and labels.

        Args:
            features: (num_instances, num_features).
            labels: (num_instances).
    """

    # Sort features
    s_idx = np.argsort(labels)
    labels = labels[s_idx]
    features = features[s_idx, :]

    class_mat = pairwise_distances(labels.reshape(-1, 1), labels.reshape(-1, 1))
    class_mat[np.eye(class_mat.shape[0]) == 1] = -1  # mark own_sample relations
    class_mat[class_mat > 0] = 1  # mark inter_class relations

    # Calculate pairwise distances matrix
    pd_mat = squareform(pdist(features))

    intra_class_dist = pd_mat[class_mat == 0]
    inter_class_dist = pd_mat[class_mat == 1]

    return intra_class_dist, inter_class_dist



