import os
import torch
import matplotlib.pyplot as plt

log_dir = 'logs_orig'
resume = 'epoch_95.pth.tar'

state_file = os.path.join(log_dir, 'models', resume)
if not os.path.isfile(state_file):
    raise RuntimeError(
        "resume file {} is not found".format(state_file))
print("loading checkpoint {}".format(state_file))
checkpoint = torch.load(state_file)
start_epoch = current_epoch = checkpoint['epoch']

training_losses = checkpoint['training_losses']
validation_losses = checkpoint['validation_losses']
print("loaded checkpoint {} (epoch {})".format(
    state_file, current_epoch))

plt.plot(validation_losses['top1acc'][:], label='orig')



#######################333


log_dir = 'logs_myloss'
resume = 'epoch_95.pth.tar'

state_file = os.path.join(log_dir, 'models', resume)
if not os.path.isfile(state_file):
    raise RuntimeError(
        "resume file {} is not found".format(state_file))
print("loading checkpoint {}".format(state_file))
checkpoint = torch.load(state_file)
start_epoch = current_epoch = checkpoint['epoch']

training_losses = checkpoint['training_losses']
validation_losses = checkpoint['validation_losses']
print("loaded checkpoint {} (epoch {})".format(
    state_file, current_epoch))

plt.plot(validation_losses['top1acc'][:], label='myloss')


plt.legend()
plt.show()

