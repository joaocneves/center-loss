import os
import argparse
from math import sqrt

import torch

from torch.utils.data import DataLoader


from dataset import Dataset, create_datasets_with_attributes, LFWPairedDataset
from loss import compute_center_loss, get_center_delta
from models import Resnet50FaceModel, Resnet18FaceModel
from device import device
from utils.aux_fun_inference import extract_features, extract_features_with_attributes
from utils.metrics.consistency_metric import consistency_metric_distance, consistency_metric_rank
from utils_fun import download, generate_roc_curve, image_loader
from metrics import compute_roc, select_threshold
from imageaug import transform_for_infer, transform_for_training
from utils.metrics.lfw_evaluation import lfw_evaluation_protocol


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




def evaluate_metrics(model, test_set_name, test_set_path, attributes_file, batch_size=100, device='cpu'):

    embeddings, attributes, targets = extract_features_with_attributes(model, test_set_name, test_set_path, attributes_file, batch_size, device)
    thresholds, avg_dist_global = consistency_metric_distance(embeddings, attributes)
    ranks, att_dist_rank = consistency_metric_rank(embeddings, attributes)

    metrics_distance = {'thresholds': thresholds, 'avg_dist_global': avg_dist_global}
    metrics_rank = {'ranks': ranks, 'att_dist_rank': att_dist_rank}


    return metrics_distance, metrics_rank

def evaluate_model(model, test_set_path, pairs_file, batch_size=100):
    dataset_dir = test_set_path

    eval_flip_images = False
    nrof_flips = 2 if eval_flip_images else 1

    pairs_path = pairs_file if pairs_file else \
        os.path.join(dataset_dir, 'pairs.txt')

    if not os.path.isfile(pairs_path):
        download(dataset_dir, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    dataset = LFWPairedDataset(
        dataset_dir, pairs_path, transform_for_infer(model.IMAGE_SHAPE))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=6)

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

    _acc = dict({'nol2_euc': None, 'nol2_cos': None, 'l2_euc': None, 'l2_cos': None})
    _statstest = dict({'nol2_euc': None, 'nol2_cos': None, 'l2_euc': None, 'l2_cos': None})
    _statstrain = dict({'nol2_euc': None, 'nol2_cos': None, 'l2_euc': None, 'l2_cos': None})
    _acc['nol2_euc'], _statstest['nol2_euc'], _statstrain['nol2_euc'] = \
        lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=False, dist='euclidean')
    _acc['l2_euc'], _statstest['l2_euc'], _statstrain['l2_euc'] = \
        lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=True, dist='euclidean')
    _acc['nol2_cos'], _statstest['nol2_cos'], _statstrain['nol2_cos'] = \
        lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=False, dist='cosine')
    _acc['l2_cos'], _statstest['l2_cos'], _statstrain['l2_cos'] = \
        lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=True, dist='cosine')

    return _acc, _statstest, _statstrain


def evaluate(args):
    checkpoint = torch.load(args.weights)
    num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
    model_class = get_model_class(args)

    model = model_class(num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    accuracy, stats_test, stats_train = evaluate_model(model, args.test_dataset_dir, args.pairs_file, batch_size=args.batch_size)

    return accuracy, stats_test, stats_train

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
    evaluate(args)