import os

import PIL
import numpy as np
import torch
import cv2
from eval import eval
import utils.aux_fun_inference as aux

""" 
This script was used to determine the accuracy of lfw verification task when using the features provided by face_insight
It confirms that our evaluation scripts are good because the acc=0.995 (the same value reported)
"""


# img = np.load('/home/socialab/Desktop/Joao/projects/InsightFace_Pytorch/test.npy')
# im =  PIL.Image.fromarray((img*0.5+0.5)*255)
# im.show()

os.mkdir('tmp_data_lfw_embed_insight')

DATASET_DIR = '/home/socialab/Desktop/Joao/datasets/lfw-aligned_insight'
image_size = (112,112)
images, targets = aux.create_lfw_bin_all_images(DATASET_DIR, image_size)
np.save('tmp_data_lfw_embed_insight/lfw_all_imgs.npy', images)
np.save('lfw_all_targets.npy', targets)


DATASET_DIR = '/home/socialab/Desktop/Joao/datasets/lfw-aligned_insight'
pairs_path = '../datasets/lfw/pairs.txt'
image_size = (112,112)
batch_size = 100
imgs = create_lfw_bin(DATASET_DIR, pairs_path, image_size, batch_size)
np.save('../lfw_emore_replica.npy', imgs)
im =  PIL.Image.fromarray(((imgs.cpu().numpy())[0,0,:,:]*0.5+0.5)*255)
im.show()



embeddings = np.load('/home/socialab/Desktop/Joao/projects/InsightFace_Pytorch/embedings_lfw.npy')
matches = np.load('/home/socialab/Desktop/Joao/projects/InsightFace_Pytorch/matches_lfw.npy')
matches = torch.tensor(matches)

embeddings_a = torch.tensor(embeddings[0::2])
embeddings_b = torch.tensor(embeddings[1::2])

accuracy, stats_test, stats_train = eval(embeddings_a, embeddings_b, matches)

print(accuracy)