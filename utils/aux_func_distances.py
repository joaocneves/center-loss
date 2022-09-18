import random

import numpy as np

import torch
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import pairwise_distances



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



