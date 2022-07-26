import random

import numpy as np
import seaborn as sns
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

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


# -------------------------------------------
# CALCULATE THE INTRA/INTER CLASS DISTANCES
# -------------------------------------------

# data = np.load('epoch_95.npz')
# features = data['features']
# labels = data['labels']
#
# intra_class_dist, inter_class_dist = intra_inter_distances(features, labels)
# inter_class_dist = np.random.choice(inter_class_dist, size=100000)
#
# sns.kdeplot(intra_class_dist, bw=0.5)
# sns.kdeplot(inter_class_dist, bw=0.5)
# plt.show()

# -------------------------------------------
# CALCULATE THE INTRA/INTER SOFT TRAIT DISTANCES
# -------------------------------------------

features = np.load('all_features.npy')
data = np.load('epoch_2.npz')
features = data['features']
labels = data['labels']

soft_feat = np.load('datasets\\lfw\\atts_lfw.npy')
soft_feat = soft_feat[labels]
soft_feat = np.around(soft_feat, 0)
soft_feat = soft_feat[:,1]

mask = (soft_feat==1) | (soft_feat==4)
features = features[mask,:]
soft_feat = soft_feat[mask]

intra_class_dist, inter_class_dist = intra_inter_distances(features, soft_feat)
intra_class_dist = np.random.choice(intra_class_dist, size=100000)
inter_class_dist = np.random.choice(inter_class_dist, size=100000)

sns.kdeplot(intra_class_dist, bw=0.5)
sns.kdeplot(inter_class_dist, bw=0.5)
plt.show()
g = 1
