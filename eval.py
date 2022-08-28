import os
import argparse
from math import sqrt

import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset import Dataset, create_datasets, LFWPairedDataset
from loss import compute_center_loss, get_center_delta
from models import Resnet50FaceModel, Resnet18FaceModel
from device import device
from utils import download, generate_roc_curve, image_loader
from metrics import compute_roc, select_threshold
from imageaug import transform_for_infer, transform_for_training

def myroccurve(y_true, y_score, thresholds):

    tpr = np.zeros((len(thresholds),1))
    fpr = np.zeros((len(thresholds),1))

    for threshold_index, threshold in enumerate(thresholds):
        predicts = y_score < threshold

        tp = torch.sum(predicts & y_true).item()
        fp = torch.sum(predicts & ~y_true).item()
        tn = torch.sum(~predicts & ~y_true).item()
        fn = torch.sum(~predicts & y_true).item()

        tpr[threshold_index] = float(tp) / (tp + fn)
        fpr[threshold_index] = float(fp) / (fp + tn)


    return tpr, fpr

def get_dataset_dir(args):
    home = os.path.expanduser("~")
    dataset_dir = args.dataset_dir if args.dataset_dir else os.path.join(
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


def get_model_class(args):
    if args.arch == 'resnet18':
        model_class = Resnet18FaceModel
    if args.arch == 'resnet50':
        model_class = Resnet50FaceModel
    elif args.arch == 'inceptionv3':
        model_class = InceptionFaceModel

    return model_class


def optimal_threshold(fpr, tpr, th):
    th_score = np.sqrt((tpr - 1) ** 2 + fpr ** 2)
    optim_th = np.argmin(th_score)

    return th[optim_th]


def eval(embedings_a, embedings_b, matches):
    nfolds = 10  # predfined by LFW protocol

    stats_train = dict.fromkeys(['tpr', 'fpr', 'auc'])
    stats_test = dict.fromkeys(['tpr', 'fpr', 'auc'])

    fpr_train = [None]*nfolds
    tpr_train = [None]*nfolds
    auc_train = np.zeros((nfolds, 1))
    fpr_test = [None]*nfolds
    tpr_test = [None]*nfolds
    auc_test = np.zeros((nfolds, 1))
    accuracy = np.zeros((nfolds, 1))

    folds = [list(range(600 * i, 600 * (i + 1))) for i in range(10)]
    folds = np.array(folds)

    scores = -torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1).cpu()
    thresholds = np.unique(scores.cpu())
    #thresholds = np.arange(0, 4, 0.05)
    #distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)
    #tpr, fpr, accuracy, best_thresholds = compute_roc(distances, matches, thresholds, fold_size=10)

    for i in range(10):
        test_fold = i
        train_fold = list(set(list(range(10))) - set([i]))
        train_fold = np.array(train_fold)

        test_idx = folds[test_fold]
        train_idx = folds[train_fold].ravel()


        scores_train = -torch.sum(torch.pow(embedings_a[train_idx] - embedings_b[train_idx], 2), dim=1).cpu()
        matches_train = matches[train_idx].cpu()
        fpr_train[i], tpr_train[i] = myroccurve(matches_train, scores_train, thresholds)
        auc_train[i] = roc_auc_score(matches_train, scores_train)
        optim_th = optimal_threshold(fpr_train[i], tpr_train[i], thresholds)

        scores_test = -torch.sum(torch.pow(embedings_a[test_idx] - embedings_b[test_idx], 2), dim=1).cpu()
        matches_test = matches[test_idx].cpu()
        fpr_test[i], tpr_test[i] = myroccurve(matches_test, scores_test, thresholds)
        auc_test[i] = roc_auc_score(matches_test, scores_test)

        true_predicts = torch.sum((scores_test > optim_th) == matches_test)
        accuracy[i] = true_predicts.item() / float(len(test_idx))

    # average fold

    tpr_train = np.hstack(tpr_train)
    fpr_train = np.hstack(fpr_train)
    tpr_test = np.hstack(tpr_test)
    fpr_test = np.hstack(fpr_test)

    stats_train['tpr'] = np.mean(tpr_train, axis=1)
    stats_train['fpr'] = np.mean(fpr_train, axis=1)
    stats_train['auc'] = np.mean(auc_train)
    stats_test['tpr'] = np.mean(tpr_test, axis=1)
    stats_test['fpr'] = np.mean(fpr_test, axis=1)
    stats_test['auc'] = np.mean(auc_test)
    avg_accuracy = np.mean(accuracy)



    return avg_accuracy, stats_test, stats_train


def evaluate(model, test_set_path, pairs_file, batch_size=1):
    dataset_dir = test_set_path

    eval_flip_images = False
    nrof_flips = 2 if eval_flip_images else 1

    pairs_path = pairs_file if pairs_file else \
        os.path.join(dataset_dir, 'pairs.txt')

    if not os.path.isfile(pairs_path):
        download(dataset_dir, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    dataset = LFWPairedDataset(
        dataset_dir, pairs_path, transform_for_infer(model.IMAGE_SHAPE))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    model.eval()

    embedings_a = torch.zeros(len(dataset), model.FEATURE_DIM * nrof_flips)
    embedings_b = torch.zeros(len(dataset), model.FEATURE_DIM * nrof_flips)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) \
            in enumerate(dataloader):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        _, batched_embedings_a = model(images_a)
        _, batched_embedings_b = model(images_b)

        if eval_flip_images:
            images_a = torch.fliplr(images_a)
            images_b = torch.fliplr(images_b)

            _, batched_embedings_a_flip = model(images_a)
            _, batched_embedings_b_flip = model(images_b)

        start = batch_size * iteration
        end = start + current_batch_size

        # embedings_a[start:end, :] = batched_embedings_a.data
        # embedings_b[start:end, :] = batched_embedings_b.data

        embedings_a[start:end, 0:model.FEATURE_DIM] = batched_embedings_a.data
        embedings_b[start:end, 0:model.FEATURE_DIM] = batched_embedings_b.data
        if eval_flip_images:
            embedings_a[start:end, 0:model.FEATURE_DIM] = batched_embedings_a.data + batched_embedings_a_flip.data
            embedings_b[start:end, 0:model.FEATURE_DIM] = batched_embedings_b.data + batched_embedings_b_flip.data

        matches[start:end] = batched_matches.data

    accuracy, stats_test, stats_train = eval(embedings_a, embedings_b, matches)

    print('Model accuracy is {}'.format(accuracy))


def evaluate_model(args):
    checkpoint = torch.load(args.weights)
    num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
    model_class = get_model_class(args)

    model = model_class(num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    evaluate(model, args.test_dataset_dir, args.pairs_file, batch_size=args.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='center loss example')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--log_dir', type=str,
                        help='log directory')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='network arch to use, support resnet18 and '
                             'resnet50 (default: resnet50)')
    parser.add_argument('--test_dataset_dir', type=str,
                        help='directory with lfw dataset'
                             ' (default: $HOME/datasets/lfw)')
    parser.add_argument('--weights', type=str,
                        help='pretrained weights to load '
                             'default: ($LOG_DIR/resnet18.pth)')
    parser.add_argument('--pairs_file', type=str,
                        help='path of pairs.txt '
                             '(default: $DATASET_DIR/pairs.txt)')

    args = parser.parse_args()
    evaluate_model(args)