import os
import numpy as np

for att_loss_weight in np.arange(0.0, 1.0, 0.1):

    command = 'python3 main.py --experiment_name imgsize_112_attweight_' + str(att_loss_weight) + ' '\
              '--loss attribute-loss ' \
              '--epochs 40 ' \
              '--batch_size 128 ' \
              '--train_dataset_dir datasets/celeba/celeba-aligned_insight ' \
              '--test_dataset_dir /home/socialab/Joao/datasets/lfw-aligned_insight ' \
              '--pairs datasets/lfw/pairs.txt ' \
              '--atributes_file datasets/celeba/atts_celeba.npy ' \
              '--att_loss_weight ' + str(att_loss_weight)
    print(command)
    os.system(command)