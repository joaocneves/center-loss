import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.aux_func_distances import intra_inter_distances

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
