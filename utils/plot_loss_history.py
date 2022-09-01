import os
import torch
import matplotlib.pyplot as plt

log_dir = '../logs/attribute-loss.imgsize_112_attweight_0.0'
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
plt.plot(lfw_acc, label='orig')


#######################


log_dir = '../logs/attribute-loss.img_size_112'
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
plt.plot(lfw_acc, label='myloss')

plt.legend()
plt.show()

