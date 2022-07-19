import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

soft_feat = \
            [[1, 1, 1, 0, 1, 1, 1],
             [0, 0, 0, 0, 0, 1, 1],
             [0, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1],
             [1, 0, 0, 1, 0, 1, 1],
             [1, 0, 1, 1, 1, 0, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 0, 1, 0, 0, 1, 1],
             [1, 1, 1, 1, 1, 1, 1],
             [1, 0, 1, 1, 1, 1, 1]]

soft_feat = np.array(soft_feat).astype('float32')
labels = np.arange(0,10)

num_classes = 10


dist_mat = squareform(pdist(soft_feat))
mds = MDS(dissimilarity='precomputed', random_state=0)
X_transform_L1 = mds.fit_transform(dist_mat)


colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
for label_idx in range(num_classes):
    plt.scatter(
        X_transform_L1[labels == label_idx, 0],
        X_transform_L1[labels == label_idx, 1],
        c=colors[label_idx],
        s=30,
    )
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
plt.show()