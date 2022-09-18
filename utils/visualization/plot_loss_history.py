import os

import numpy as np
import torch
import matplotlib.pyplot as plt

log_dir = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs/center-loss.imgsize_112'
#resume = 'epoch_14.pth.tar'
resume = 'stats.pt'

state_file = os.path.join(log_dir, resume)
if not os.path.isfile(state_file):
    raise RuntimeError(
        "resume file {} is not found".format(state_file))
print("loading checkpoint {}".format(state_file))
checkpoint = torch.load(state_file)


#plt.plot(checkpoint['test_stats']['fpr'][0], checkpoint['test_stats']['tpr'][0])
#plt.plot(checkpoint['test_stats']['fpr'][16], checkpoint['test_stats']['tpr'][16])
#plt.show()


training_losses = checkpoint['train_stats']
validation_losses = checkpoint['val_stats']
lfw_acc = checkpoint['test_stats']['acc']['l2_cos']

#plt.plot(validation_losses['top1acc'][:], label='orig')
plt.plot(lfw_acc, label='orig', linestyle = 'dashed')
#plt.show()

#######################

for att_loss_weight in np.arange(0.5, 0.6, 0.2):

    att_loss_weight = np.round(att_loss_weight,2)

    if att_loss_weight ==1.1:
        continue
    log_dir = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/logs/attribute-loss.imgsize_112'
    #resume = 'epoch_66.pth.tar'
    resume = 'stats.pt'

    state_file = os.path.join(log_dir, resume)
    if not os.path.isfile(state_file):
        raise RuntimeError(
            "resume file {} is not found".format(state_file))
    print("loading checkpoint {}".format(state_file))
    checkpoint = torch.load(state_file)
    #start_epoch = current_epoch = checkpoint['epoch']

    training_losses = checkpoint['train_stats']
    validation_losses = checkpoint['val_stats']
    lfw_acc = checkpoint['test_stats']['acc']['l2_cos']

    #plt.plot(validation_losses['top1acc'][:], label='myloss')
    plt.plot(lfw_acc, label='myloss_'+str(att_loss_weight))

plt.legend()
plt.show()

