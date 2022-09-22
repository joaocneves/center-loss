import os

import numpy as np
import torch

from utils.aux_create_sets import create_lfw_blufr_gallery_probes_sets
from utils.aux_misc import create_imgarray_from_set
from utils.metrics.lfw_blufr_protocol import cumulative_matching_curve, DIR_FAR

dataset_name = 'lfw'
DATASET_DIR = '/home/socialab/Joao/datasets/lfw-aligned_insight'
arch = 'resnet50'
weights_file = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs/attribute-loss.imgsize_112/epoch_22.pth.tar'
device = 'cuda:0'
attributes_file = '/home/socialab/Joao/projects/center_loss_main/datasets/lfw/attributes_per_person_lfw.json'

#gallery_set, probes_set = create_gallery_probe_datasets(DATASET_DIR)
images_root = '/home/socialab/Joao/datasets/lfw-aligned_insight'
blufr_lfw_config_file = '../datasets/lfw/blufr_lfw_config.mat'

EVAL = True


gallery_set, probes_set = create_lfw_blufr_gallery_probes_sets(images_root, blufr_lfw_config_file, closed_set=False)

gimages, gtargets = create_imgarray_from_set(gallery_set, (112, 112))
pimages, ptargets = create_imgarray_from_set(probes_set, (112, 112))

if not EVAL:

    imgsarray = np.vstack([gimages, pimages])

    #os.mkdir('tmp_data_lfw_embed_insight')
    #np.save('tmp_data_lfw_embed_insight/imgsarray_lfw_blufr.npy', imgsarray)
else:

    embeddings = np.load('tmp_data_lfw_embed_insight/embeddings_lfw_blufr.npy')
    gembeddings = torch.tensor(embeddings[:1000, :])
    pembeddings = torch.tensor(embeddings[1000:, :])

    #gtargets = torch.tensor(gtargets)
    #ptargets = torch.tensor(ptargets)

    DIR, FAR = DIR_FAR(pembeddings, ptargets, gembeddings, gtargets, dist='l2')
print()