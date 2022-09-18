import os
import numpy as np


command = 'python3 main.py --experiment_name imgsize_112_centerloss '\
              '--loss center-loss ' \
              '--epochs 40 ' \
              '--batch_size 128 ' \
              '--log_dir /media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs ' \
              '--train_dataset_dir datasets/celeba/celeba-aligned_insight ' \
              '--test_dataset_dir /home/socialab/Joao/datasets/lfw-aligned_insight ' \
              '--pairs datasets/lfw/pairs.txt ' \
              '--atributes_file datasets/celeba/atts_celeba.npy '
print(command)
os.system(command)


for att_loss_weight in np.arange(0.1, 2.0, 0.2):

    command = 'python3 main.py --experiment_name imgsize_112_attweight_' + str(att_loss_weight) + ' '\
              '--loss attribute-loss ' \
              '--epochs 40 ' \
              '--batch_size 128 ' \
              '--log_dir /media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs ' \
              '--train_dataset_dir datasets/celeba/celeba-aligned_insight ' \
              '--test_dataset_dir /home/socialab/Joao/datasets/lfw-aligned_insight ' \
              '--pairs datasets/lfw/pairs.txt ' \
              '--atributes_file datasets/celeba/atts_celeba.npy ' \
              '--att_loss_weight ' + str(att_loss_weight)
    print(command)
    os.system(command)
