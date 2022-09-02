import json
import os
import argparse
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset import Dataset, create_datasets, LFWPairedDataset
from loss import compute_center_loss, get_center_delta
from models import Resnet50FaceModel, Resnet18FaceModel
from device import device
from trainer import Trainer
from utils import download, generate_roc_curve, image_loader
from metrics import compute_roc, select_threshold
from imageaug import transform_for_infer, transform_for_training

# Seed
seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main(args):
    train(args)


def get_dataset_dir(args):
    home = os.path.expanduser("~")
    dataset_dir = args.train_dataset_dir if args.train_dataset_dir else os.path.join(
        home, 'datasets', 'lfw')

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    return dataset_dir


def get_log_dir(args):
    log_dir = args.log_dir if args.log_dir else os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logs')

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    return log_dir

def get_experiment_name(args, log_dir):

    experiment_name = args.loss + '.' + args.experiment_name

    experiment_dir = os.path.join(log_dir, experiment_name)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return experiment_name

def get_model_class(args):
    if args.arch == 'resnet18':
        model_class = Resnet18FaceModel
    if args.arch == 'resnet50':
        model_class = Resnet50FaceModel
    elif args.arch == 'inceptionv3':
        model_class = InceptionFaceModel

    return model_class


def train(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)
    experiment_name = get_experiment_name(args, log_dir)

    training_set, validation_set, num_classes = create_datasets(dataset_dir)

    training_dataset = Dataset(
            training_set, transform_for_training(model_class.IMAGE_SHAPE))
    validation_dataset = Dataset(
        validation_set, transform_for_infer(model_class.IMAGE_SHAPE))

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False
    )

    soft_feat = np.load(args.atributes_file)
    soft_feat = torch.tensor(soft_feat.astype('float32'))
    soft_feat = soft_feat.to(device)

    model = model_class(num_classes).to(device)

    trainables_wo_bn = [param for name, param in model.named_parameters() if
                        param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if
                          param.requires_grad and 'bn' in name]

    optimizer = torch.optim.Adam([
        {'params': trainables_wo_bn},
        {'params': trainables_only_bn}
    ], lr=args.lr)#, weight_decay=0.0005)

    trainer = Trainer(
        optimizer,
        model,
        args.loss,
        training_dataloader,
        validation_dataloader,
        soft_feat=soft_feat,
        test_set_path=args.test_dataset_dir,
        max_epoch=args.epochs,
        resume=args.resume,
        log_dir=log_dir,
        experiment_name=experiment_name,
        att_loss_weight=args.att_loss_weight
    )
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='center loss example')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--log_dir', type=str,
                        help='log directory')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='network arch to use, support resnet18 and '
                             'resnet50 (default: resnet50)')
    parser.add_argument('--loss', type=str, default='softmax',
                        help='loss type to train the model (default: softmax)')
    parser.add_argument('--resume', type=str,
                        help='model path to the resume training',
                        default=False)
    parser.add_argument('--train_dataset_dir', type=str,
                        help='directory with lfw dataset'
                             ' (default: $HOME/datasets/casiawebface)')
    parser.add_argument('--test_dataset_dir', type=str,
                        help='directory with lfw dataset'
                             ' (default: $HOME/datasets/lfw)')
    parser.add_argument('--atributes_file', type=str,
                        default='datasets/lfw/atts_lfw.npy')
    parser.add_argument('--weights', type=str,
                        help='pretrained weights to load '
                             'default: ($LOG_DIR/resnet18.pth)')
    parser.add_argument('--pairs', type=str,
                        help='path of pairs.txt '
                             '(default: $DATASET_DIR/pairs.txt)')
    parser.add_argument('--att_loss_weight', type=float, default=0.5,
                        help='verify 2 images of face belong to one person,'
                             'split image pathes by comma')

    args = parser.parse_args()
    main(args)
