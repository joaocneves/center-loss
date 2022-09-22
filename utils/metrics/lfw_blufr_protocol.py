import os

import numpy as np
import scipy as sp
import torch

from utils.aux_create_sets import create_lfw_blufr_gallery_probes_sets
from sklearn.preprocessing import normalize as l2norm
from utils.aux_func_distances import fast_matrix_rowcol_distance

def DIR_FAR(probe_embeddings, probe_ids, gallery_embeddings, gallery_ids, dist='cosine'):

    rankPoints = np.array(range(20))

    """ Calculate Probe/Gallery Scores"""
    if dist =='l2':
        prob_gal_dist = fast_matrix_rowcol_distance(l2norm(probe_embeddings,axis=1), l2norm(gallery_embeddings,axis=1))
    elif dist =='cosine':
        prob_gal_dist = sp.distance.cdist(l2norm(probe_embeddings,axis=1), l2norm(gallery_embeddings,axis=1), metric='cosine')

    prob_gal_scores = 1- prob_gal_dist


    """ Separate Genuine/Imposter Probes"""
    # determine whether a probe belongs to the gallery
    p_in_gal = np.array([1 if probe_id in gallery_ids else 0 for probe_id in probe_ids ])

    gen_prob_index = np.where(p_in_gal == 1)[0]
    imp_prob_index = np.where(p_in_gal == 0)[0]

    ngen = len(gen_prob_index)
    nimp = len(imp_prob_index)

    """ Build FAR array and thresholds """

    false_alarms = np.array(range(nimp))
    FAR = false_alarms / nimp

    imp_scores = np.array([np.max(prob_gal_scores[i, :]) for i in imp_prob_index])
    imp_scores_sorted = np.sort(imp_scores)[::-1] # sort descending

    # the scores of the descending imposters determine the thresholds to achieve the FARs in variable FAR
    thresholds = imp_scores_sorted
    # thresholds[0] represents the threshold to obtain FAR = 0
    # since the decision is made by the ">=" operator, the decision threshold should be a bit larger than impScore[0]
    thresholds[0] = imp_scores_sorted[0] + np.finfo(np.float32).eps



    """ Build Ranks of the Genuine Probes """
    prob_gal_scores_genunine = prob_gal_scores.take(gen_prob_index, axis=0)
    prob2gal_scores_gen = np.max(prob_gal_scores_genunine, axis=1)
    prob_matches_by_score_gen = torch.argsort(torch.tensor(prob_gal_scores_genunine), dim=1, descending=True).detach().cpu().numpy()

    gallery_ids_mat = np.tile(gallery_ids, [ngen, 1])
    #prob_ids_mat = np.tile(probe_ids, [ngal, 1]).T
    # sort gallery id by rankings
    gallery_ids_rank_mat = np.take_along_axis(gallery_ids_mat, prob_matches_by_score_gen, axis=1)
    probe_ids_gen = [probe_ids[i] for i in gen_prob_index]
    prob_ranks = [ np.where(gallery_ids_rank_mat[i,:]==prob_id)[0][0] for i, prob_id in enumerate(probe_ids_gen)]

    prob_gal_correct_match_score = [prob_gal_scores_genunine[i][np.where(gallery_ids_mat[i, :] == prob_id)[0][0]] for i, prob_id in enumerate(probe_ids_gen)]

    #prob_gal_scores_genunine_sorted = np.take_along_axis(prob_gal_scores_genunine, prob_matches_by_score_gen, axis=1)
    #prob_gal_correct_match_score =  [ prob_gal_scores_genunine_sorted[i,rank]  for i, rank in enumerate(prob_ranks)]

    # thresholds[-1] represents the threshold to obtain FAR = 1, i.e., accepted every imposter
    # so the decision threshold should be the minimum score that can also accept all genuine scores
    thresholds[-1] = min(imp_scores[-1], min(prob_gal_correct_match_score)) - np.finfo(np.float32).eps

    T1 = np.array([prob_gal_correct_match_score >= t for t in thresholds]).T
    T2 = np.array([prob_ranks <= rank_p for rank_p in rankPoints]).T

    T1_full = np.repeat(T1[:, :, np.newaxis], len(rankPoints), axis=2)
    T2_full = T2[:, np.newaxis, :]
    T2_full = np.repeat(T2_full, len(thresholds), axis=1)

    T = np.logical_and(T1_full, T2_full)

    # T = np.zeros((ngen, len(thresholds), len(rankPoints)))
    # p_th_rank_combs = itertools.product(range(ngen), range(len(thresholds)), range(len(rankPoints)))
    # for ijk in p_th_rank_combs:
    #     i = ijk[0]
    #     j = ijk[1]
    #     k = ijk[2]
    #     if T1[i][j] and T2[k][i]:
    #         T[i,j,k] = 1

    DIR = np.average(T.astype(int), axis=0)

    return FAR, DIR

def cumulative_matching_curve(probe_embeddings, probe_ids, gallery_embeddings, gallery_ids, dist='l2'):
    npr = len(probe_ids)
    ngal = len(gallery_ids)

    if dist == 'l2':
        prob_embs_dist = fast_matrix_rowcol_distance(l2norm(probe_embeddings, axis=1), l2norm(gallery_embeddings, axis=1))
    elif dist == 'cosine':
        prob_embs_dist = sp.distance.cdist(probe_embeddings, gallery_embeddings, metric='cosine')
    # prob_embs_dist_2 = scipy.spatial.distance_matrix(probe_embeddings, gallery_embeddings)

    ranks = torch.argsort(torch.tensor(prob_embs_dist), dim=1).detach().cpu().numpy()

    gallery_ids_mat = np.tile(gallery_ids, [npr, 1])  # matrix containing the row-wise gallery ids repeated each row
    probes_ids_mat = np.tile(probe_ids,
                             [ngal, 1]).T  # matrix containing the column-wise probes ids repeated each column
    """
    Example for 2 probes and 20 galleries
    gallery_ids_mat = 
    [1, 2, 3, ..., 20,
     1, 2, 3

    probes_ids_mat (npr, ngal)= 
    [3, 3, ..., 3
     7, 7, ..., 7]
    """

    # sort gallery id by rankings
    gallery_ids_rank_mat = np.take_along_axis(gallery_ids_mat, ranks, axis=1)

    matches = gallery_ids_rank_mat == probes_ids_mat
    matches = matches.astype(int)

    cum_match = np.cumsum(matches, axis=1)
    cmc = np.average(cum_match, axis=0)

    return cmc


if __name__ == '__main__':

    images_root = '/home/socialab/Joao/datasets/lfw-aligned_insight'
    blufr_lfw_config_file = '../../datasets/lfw/blufr_lfw_config.mat'
    gallery_set, probe_set = create_lfw_blufr_gallery_probe_sets(images_root, blufr_lfw_config_file, closed_set=True)