import itertools
import os

import numpy as np
import scipy.spatial as sp
import sklearn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import normalize as l2norm

from dataset import create_gallery_probe_datasets
from utils.aux_fun_inference import extract_features, extract_features_identification, \
    extract_features_lfw_blufr_identification
from utils.aux_fun_model import load_model
from utils.aux_func_distances import fast_pairwise_dist
from utils.metrics.lfw_blufr_protocol import create_lfw_blufr_gallery_probes_sets
from utils_fun import image_loader









def get_gallery_ranking(probe_embedding, gallery_embeddings, distance='l2', device='cpu'):
    if distance == 'l2':
        emb_dist = \
            torch.nn.functional.pairwise_distance(torch.tensor(probe_embedding).to(device),
                                                  torch.tensor(gallery_embeddings).to(device))
    elif distance == 'cosine':
        emb_dist = \
            torch.nn.functional.cosine_similarity(torch.tensor(probe_embedding).to(device),
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
weights_file = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs/attribute-loss.imgsize_112/epoch_22.pth.tar'
device = 'cuda:0'
attributes_file = '/home/socialab/Joao/projects/center_loss_main/datasets/lfw/attributes_per_person_lfw.json'

#gallery_set, probes_set = create_gallery_probe_datasets(DATASET_DIR)
images_root = '/home/socialab/Joao/datasets/lfw-aligned_insight'
blufr_lfw_config_file = '../../datasets/lfw/blufr_lfw_config.mat'

gallery_set, probes_set = create_lfw_blufr_gallery_probes_sets(images_root, blufr_lfw_config_file, closed_set=False)
model = load_model(arch, weights_file, device)
gallery_features_mat, gallery_targets_mat, probe_features_mat, probe_targets_mat = \
    extract_features_lfw_blufr_identification(model, DATASET_DIR, blufr_lfw_config_file, batch_size=128, device='cuda:0')

probe_img_list = []
gallery_match_list = []
for i in range(610, 625):
    probe_embedding = probe_features_mat[i, :]
    ranks = get_gallery_ranking(probe_embedding, gallery_features_mat, distance='cosine')

    probe_img_list.append(probes_set[i])
    gallery_match_list.append([gallery_set[i] for i in ranks[:7]])


detection(probe_features_mat, probe_targets_mat, gallery_features_mat, gallery_targets_mat, dist='l2')
cmc = cumulative_matching_curve(probe_features_mat, probe_targets_mat, gallery_features_mat, gallery_targets_mat, dist='l2')
#create_grid_rank(probe_img_list, gallery_match_list)
print('p')
