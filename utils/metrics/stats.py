from math import sqrt

import numpy as np
from scipy.spatial.distance import squareform, pdist

def relative_distance_histogram(soft_feat, hard_feat):

    soft_feat = np.around(soft_feat, 0)
    sf_dist_mat = squareform(pdist(soft_feat))
    sf_dist_mat = np.around(sf_dist_mat, 2)
    unique_dists = np.unique(sf_dist_mat)
    unique_dists = unique_dists[1:]  # remove the zero distance

    hf_dist_mat = squareform(pdist(hard_feat))/sqrt(hard_feat.shape[1])

    n = len(unique_dists)
    samp_per_unique_dist = []
    for i in range(n):
        samp_per_unique_dist.append([])

    for i in range(soft_feat.shape[0]):
        for j in range(soft_feat.shape[0]):

            # calculate the difference of the this distance to the different clusters (unique_dists)
            dclst = np.abs(unique_dists - sf_dist_mat[i][j])
            i_dclst = np.argmin(dclst)
            # check if the distance is in the [d*0.9, d*1.1] range
            if (dclst[i_dclst] < 0.1 * unique_dists[i_dclst]):
                samp_per_unique_dist[i_dclst].append(hf_dist_mat[i][j])


    return  samp_per_unique_dist


soft_feat = np.load('datasets\\lfw\\atts_lfw.npy')
soft_feat = np.around(soft_feat, 0)
soft_feat = soft_feat[:,[0]]

dist_mat = squareform(pdist(soft_feat))
dist_mat = np.around(dist_mat, 2)
unique_dists = np.unique(dist_mat)

print(soft_feat)

#all_features = np.load('all_features.npy')
#all_labels = np.load('all_labels.npy')

data = np.load('epoch_2.npz')
features = data['features']
labels = data['labels']


soft_feat_per_label = soft_feat[labels]
samp_per_unique_dist = relative_distance_histogram(soft_feat_per_label, features)

import pickle
with open("samp_per_unique_dist.pkl", "wb") as fp:   #Pickling
    pickle.dump(samp_per_unique_dist, fp)
