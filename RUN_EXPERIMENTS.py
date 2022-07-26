import os

for att_loss_weight in range(0.0, 1.0, 0.1):

    command = 'python main.py --epochs 150 --dataset_dir datasets/lfw --att_loss_weight ' + str(att_loss_weight)
    os.system(command)