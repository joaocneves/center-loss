import numpy as np
import torch

from device import device


def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def compute_relative_loss(features, centers, targets, soft_feat, person_attributes):

    batch_size = features.size(0)
    num_classes = centers.size(0)
    soft_feat_per_label = soft_feat[targets]
    # calculate pairwise distance between soft_feat_per_label and soft_feat
    relative_dist_mat = torch.pow(soft_feat_per_label, 2).sum(dim=1, keepdim=True).expand(batch_size,
                                                                                          num_classes) + \
                        torch.pow(soft_feat, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
    relative_dist_mat.addmm_(1, -2, soft_feat_per_label, soft_feat.t())

    relative_dist_mat = relative_dist_mat/soft_feat.size(1)

    # calculate pairwise distance between features and centers
    distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
              torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
    distmat.addmm_(1, -2, features, centers.t())
    distmat = distmat / features.size(1)

    #dist = torch.pow(torch.pow(distmat - relative_dist_mat, 2), 1)
    #print(torch.max(dist))
    #loss = dist.clamp(min=1e-12, max=1e+12).sum()

    criterion = torch.nn.MSELoss()
    loss = criterion(distmat, relative_dist_mat)

    return loss


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """
    beta = torch.tensor(beta).to('cuda:0')
    # Compute P-row and corresponding perplexity
    P = torch.exp(-D.clone() * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    #print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = torch.sum(torch.square(X), 1)
    D = torch.add(torch.add(-2 * torch.matmul(X, X.T), sum_X).T, sum_X)
    P = torch.zeros((n, n)).to('cuda:0')
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        #if i % 500 == 0:
        #    print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff.cpu()) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP.float()

    # Return final P-matrix
    #print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def KL_loss(features, targets, soft_feat):

    batch_size = features.size(0)
    soft_feat_per_label = soft_feat


    # Compute P-values
    perplexity = 50
    X = soft_feat_per_label
    P = x2p(X, 1e-5, perplexity)
    P = torch.add(P, P.t())
    P = P / torch.sum(P)
    #P = P * 4.									# early exaggeration
    P = torch.clamp(P, min=1e-12)

    # Compute pairwise affinities
    Y = features
    (n, d) = X.shape
    sum_Y = torch.sum(torch.square(Y), 1)
    num = -2. * torch.matmul(Y, Y.T)
    num = 1. / (1. + torch.add(torch.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0.
    Q = num / torch.sum(num)
    Q = torch.clamp(Q, min=1e-12)

    #print(P[18,0])
    #print(Q[18,0])
    #KL = P*torch.log(P/Q)
    #print(KL[18,0])

    #cross_entropy_loss = torch.nn.functional.cross_entropy(P, Q)

    loss = torch.nn.functional.kl_div(Q.log(), P, None, None, 'sum')
    #loss = torch.nn.functional.kl_div(P.log(), Q, None, None, 'sum')
    return loss

def get_center_delta(features, centers, targets, alpha):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result
