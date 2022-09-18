import torch
import numpy as np
from utils.aux_func_distances import fast_pairwise_dist


def consistency_metric_rank(embeddings, attributes, binary=False):

    """ Calculates the average dissimilarity of the attributes of the ith nearest neighbour

    embeddings - NxA matrix
    attributes - NxB matrix
    binary (True/False) - determines if the attribute distance should be the real distance or just a binary value (equal/ not equal)

    """

    if type(embeddings) is np.ndarray:
        embeddings = embeddings.astype('float32')

    if type(attributes) is np.ndarray:
        attributes = attributes.astype('float32')

    emb_dist = fast_pairwise_dist(embeddings, 'cuda:0')
    att_dist = fast_pairwise_dist(attributes, 'cuda:0')

    if binary:
        att_dist[att_dist > 0] = 1
        att_dist = 1.0 - att_dist

    emb_argsort = torch.argsort(torch.tensor(emb_dist), dim=1).detach().cpu().numpy()

    att_dist_sort = np.take_along_axis(att_dist, emb_argsort, axis=1)

    ranks = np.array(list(range(att_dist_sort.shape[1])))
    att_dist_rank = np.cumsum(att_dist_sort, axis=1)
    att_dist_rank = np.mean(att_dist_rank, axis=0)

    return ranks, att_dist_rank


def consistency_metric_distance(embeddings, attributes, binary=False):

    """
    embeddings - NxA matrix
    attributes - NxB matrix
    binary (True/False) - determines if the attribute distance should be the real distance or just a binary value (equal/ not equal)
    """

    if type(embeddings) is np.ndarray:
        embeddings = embeddings.astype('float32')

    if type(attributes) is np.ndarray:
        attributes = attributes.astype('float32')

    emb_dist = fast_pairwise_dist(embeddings, 'cuda:0')
    att_dist = fast_pairwise_dist(attributes, 'cuda:0')

    if binary:
        att_dist[att_dist > 0] = 1
        att_dist = 1.0 - att_dist

    # since the function is not optimized the processing time grows exponentially with the number of thresholds
    # we round them to 1 decimal place to constraint the number of different thresholds
    thresholds = np.unique(np.ceil(emb_dist*2)/2)
    avg_dist_global = []

    for th in thresholds:
        neighbours_th = emb_dist <= th

        att_dist_th = np.where(neighbours_th == True, att_dist, np.nan)

        sum = np.nansum(att_dist_th, axis=1)
        count = np.sum(neighbours_th, axis=1)

        avg_dist_per_sample = sum

        avg_dist_global.append(np.average(avg_dist_per_sample))

    avg_dist_global = np.array(avg_dist_global)

    return thresholds, avg_dist_global


