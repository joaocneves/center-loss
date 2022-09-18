
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

def optimal_threshold(fpr, tpr, th):
    th_score = np.sqrt((tpr - 1) ** 2 + fpr ** 2)
    optim_th = np.argmin(th_score)

    return th[optim_th]


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


def lfw_evaluation_protocol(embedings_a, embedings_b, matches, l2_norm=True, dist='cosine'):
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

    if l2_norm:
        embedings_a = F.normalize(embedings_a, p=2, dim=1)
        embedings_b = F.normalize(embedings_b, p=2, dim=1)

    if dist == 'euclidean':
        scores = -torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1).cpu()
    else:
        scores = F.cosine_similarity(embedings_a, embedings_b).cpu()

    thresholds = np.unique(scores.cpu())
    thresholds = np.append(thresholds, thresholds[-1]+0.01)
    #thresholds = np.arange(0, 4, 0.05)
    #distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)
    #tpr, fpr, accuracy, best_thresholds = compute_roc(distances, matches, thresholds, fold_size=10)

    for i in range(10):
        test_fold = i
        train_fold = list(set(list(range(10))) - set([i]))
        train_fold = np.array(train_fold)

        test_idx = folds[test_fold]
        train_idx = folds[train_fold].ravel()

        scores_train = scores[train_idx].cpu()
        matches_train = matches[train_idx].cpu()
        fpr_train[i], tpr_train[i] = myroccurve(matches_train, scores_train, thresholds)
        auc_train[i] = roc_auc_score(matches_train, scores_train)
        optim_th = optimal_threshold(fpr_train[i], tpr_train[i], thresholds)

        scores_test = scores[test_idx].cpu()
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