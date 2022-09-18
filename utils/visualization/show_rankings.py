import itertools
import os

import numpy as np
import scipy
import torch
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import squareform

from dataset import create_gallery_probe_datasets
from utils.aux_fun_inference import extract_features, extract_features_identification
from utils.aux_func_distances import fast_pairwise_dist
from utils_fun import image_loader

def matrix_rowcol_distance(x,y):

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

        dist = x_norm + y_norm - 2.0 * np.matmul(x,y.T)
        dist = dist ** 0.5
        return dist


def cumulative_matching_curve(probe_embeddings, probe_ids, gallery_embeddings, gallery_ids):
    npr = len(probe_ids)
    ngal = len(gallery_ids)

    prob_embs_dist = matrix_rowcol_distance(probe_embeddings, gallery_embeddings)
    #prob_embs_dist_2 = scipy.spatial.distance_matrix(probe_embeddings, gallery_embeddings)

    ranks = torch.argsort(torch.tensor(prob_embs_dist), dim=1).detach().cpu().numpy()

    gallery_ids_mat = np.tile(gallery_ids, [npr, 1])  # matrix containing the row-wise gallery ids repeated each row
    probes_ids_mat = np.tile(probe_ids, [ngal,1]).T  # matrix containing the column-wise probes ids repeated each column
    """
    Example for 2 probes and 20 galleries
    gallery_ids_mat = 
    [1, 2, 3, ..., 20,
     1, 2, 3
    
    probes_ids_mat (npr, ngal)= 
    [3, 3, ..., 3
     7, 7, ..., 7]
    """

    # sort gallery id by rankings
    gallery_ids_rank_mat = np.take_along_axis(gallery_ids_mat, ranks, axis=1)

    matches = gallery_ids_rank_mat == probes_ids_mat
    matches = matches.astype(int)

    cum_match = np.cumsum(matches, axis=1)
    cmc = np.average(cum_match, axis=0)

    return cmc


def get_gallery_ranking(probe_embedding, gallery_embeddings, distance='l2', device='cpu'):
    if distance == 'l2':
        emb_dist = \
            torch.nn.functional.pairwise_distance(torch.tensor(probe_embedding).to(device),
                                                  torch.tensor(gallery_embeddings).to(device))

    emb_argsort = torch.argsort(torch.tensor(emb_dist), dim=0).detach().cpu().numpy()

    return emb_argsort


def create_grid_rank(probe_img_list, gallery_match_list):
    N = len(probe_img_list)
    M = len(gallery_match_list[0])

    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(M + 1, N),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     direction='column'
                     )

    probe_gallery_list_flat = []
    for i, sub_list in enumerate(gallery_match_list):
        probe_gallery_list_flat.append(probe_img_list[i])
        for el in sub_list:
            probe_gallery_list_flat.append(el)

    for ax, im_data in zip(grid, probe_gallery_list_flat):
        # Iterating over the grid returns the Axes.
        im_path = im_data[0]
        person_id = im_data[1]
        im = Image.open(im_path)
        ax.imshow(im)

        ax.set_axis_off()
        ax.text(0.1, 0.9, person_id, size=15, color='blue')

    plt.show()


dataset_name = 'lfw'
DATASET_DIR = '/home/socialab/Joao/datasets/lfw-aligned_insight'
arch = 'resnet50'
weights_file = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs/center-loss.imgsize_112/epoch_22.pth.tar'
device = 'cuda:0'
attributes_file = '/home/socialab/Joao/projects/center_loss_main/datasets/lfw/attributes_per_person_lfw.json'

gallery_set, probes_set = create_gallery_probe_datasets(DATASET_DIR)

gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat = \
    extract_features_identification(dataset_name, DATASET_DIR, arch, weights_file, attributes_file=None, batch_size=128,
                                    device='cuda:0')

probe_img_list = []
gallery_match_list = []
for i in range(610, 625):
    probe_embedding = probe_features_mat[i, :]
    ranks = get_gallery_ranking(probe_embedding, gallery_features_mat, distance='l2')

    probe_img_list.append(probes_set[i])
    gallery_match_list.append([gallery_set[i] for i in ranks[:7]])

cmc = cumulative_matching_curve(probe_features_mat, probe_targets_mat, gallery_features_mat, gallery_targets_mat)
#create_grid_rank(probe_img_list, gallery_match_list)
print('p')
