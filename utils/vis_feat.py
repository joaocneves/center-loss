import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE

def get_repeated_ids(labels):

    flag_found = False
    rep_idx = []
    labels.sort()
    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            rep_idx.append(i-1)
            flag_found = True
        else:
            if flag_found:
                rep_idx.append(i - 1)
                flag_found = False

    return rep_idx



#features = np.load('all_features.npy')
#labels = np.load('all_labels.npy')

data = np.load('epoch_2.npz')
features = data['features']
labels = data['labels']

rep_idx = get_repeated_ids(labels)
features = features[rep_idx,:]
labels = labels[rep_idx]


features_2d = TSNE(n_components=2, init = 'random').fit_transform(features)

num_classes = len(np.unique(labels))

colors = np.random.rand(num_classes, 3)

u_labels = np.unique(labels) # unique labels
for i, label_idx in enumerate(u_labels):
    plt.scatter(
        features_2d[labels == label_idx, 0],
        features_2d[labels == label_idx, 1],
        color=colors[i]
    )

plt.show()



soft_feat = np.load('datasets\\lfw\\atts_lfw.npy')
soft_feat = np.around(soft_feat, 0)
soft_feat = soft_feat[:,[0]]

dist_mat = squareform(pdist(soft_feat))
dist_mat = np.around(dist_mat, 2)
unique_dists = np.unique(dist_mat)
unique_dists = unique_dists[1:]  # remove the zero distance

with open("samp_per_unique_dist.pkl", 'rb') as f:
    samp_per_unique_dist = pickle.load(f)


fig, axs = plt.subplots(5, 5)

for idx, udist in enumerate(unique_dists):

    i = int(idx / 5)
    j = int(idx % 5)

    axs[i, j].hist(samp_per_unique_dist[idx])
    axs[i, j].axvline(udist, color='k', linestyle='dashed', linewidth=1)

plt.show()