import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
import pdb


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6, alpha=100):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct, dist_an.shape[0]


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6, alpha=100, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct, closest_negative.shape[0]


class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6, alpha=100, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)

            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct, closest_negative.shape[0]


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T = 3

        predict = F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.cuda(), requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


class RobustTripletLoss_DART(nn.Module):
    def __init__(self, batch_size, margin):
        super(RobustTripletLoss_DART, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, inputs, prediction, targets, true_targets, prob, threshold):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the positive and negative
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())
        is_confident = (prob >= threshold)
        dist_ap, dist_an = [], []
        cnt, loss = 0, 0
        loss_inverse = False

        for i in range(n):
            if is_confident[i]:
                pos_idx = (torch.nonzero(is_pos[i].long())).squeeze(1)
                neg_idx = (torch.nonzero(is_neg[i].long())).squeeze(1)

                random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                while random_pos_index == i:
                    random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))

                rank_neg_index = dist[i][neg_idx].argsort()
                hard_neg_index = rank_neg_index[0]
                hard_neg_index = neg_idx[hard_neg_index]

                dist_ap.append(dist[i][random_pos_index].unsqueeze(0))
                dist_an.append(dist[i][hard_neg_index].unsqueeze(0))

                if prob[random_pos_index] >= threshold and prob[hard_neg_index] >= threshold:
                    # TP-TN
                    pass

                elif prob[random_pos_index] >= threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # TP-FN
                    if is_FN:
                        tmp = rank_neg_index[1]
                        hard_neg_index_new = neg_idx[tmp]
                        j = 1
                        loop_cnt = 0
                        while prob[hard_neg_index_new] < threshold:
                            j += 1
                            tmp = rank_neg_index[j]
                            hard_neg_index_new = neg_idx[tmp]
                            loop_cnt += 1
                            if loop_cnt >= 10:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_ap[cnt] = (dist[i][random_pos_index].unsqueeze(0) +
                                        dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_an[cnt] = dist[i][hard_neg_index_new].unsqueeze(0)
                    # TP-TN
                    else:
                        pass

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] >= threshold:
                    # FP-TN
                    random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    loop_cnt = 0
                    while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt += 1
                        if loop_cnt >= 5:
                            # print("------------warning, break the death loop---------------")
                            break
                    dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)
                                    + dist[i][hard_neg_index].unsqueeze(0)) / 2
                    dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # FP-FN
                    if is_FN:
                        loss_inverse = True
                    # FP-TN
                    else:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt = 0
                        while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                            random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                            loop_cnt += 1
                            if loop_cnt >= 5:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)
                                        + dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                if loss_inverse:
                    loss += torch.clamp(dist_an[cnt] - dist_ap[cnt] + self.margin, 0)
                else:
                    loss += torch.clamp(dist_ap[cnt] - dist_an[cnt] + self.margin, 0)

                cnt += 1
                loss_inverse = False
            else:
                continue

        # compute accuracy
        if cnt == 0:
            return torch.Tensor([0.]).to(inputs.device), 0, cnt
        else:
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            correct = torch.ge(dist_an, dist_ap).sum().item()
            return loss / cnt, correct, cnt
        

class RobustTripletLoss_LCNL(nn.Module):
    def __init__(self, batch_size, margin, op_type):
        super(RobustTripletLoss_LCNL, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.op_type = op_type

    def forward(self, inputs, prediction, targets, true_targets, prob, threshold):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the positive and negative
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())
        is_confident = (prob >= threshold)
        dist_ap, dist_an = [], []
        cnt, loss = 0, 0
        loss_inverse = False

        for i in range(n):
            if is_confident[i]:
                pos_idx = (torch.nonzero(is_pos[i].long())).squeeze(1)
                neg_idx = (torch.nonzero(is_neg[i].long())).squeeze(1)

                rank_pos_index = dist[i][pos_idx].argsort()
                hard_pos_index = rank_pos_index[-1]
                hard_pos_index = pos_idx[hard_pos_index]

                rank_neg_index = dist[i][neg_idx].argsort()
                hard_neg_index = rank_neg_index[0]
                hard_neg_index = neg_idx[hard_neg_index]

                dist_ap.append(dist[i][hard_pos_index].unsqueeze(0))
                dist_an.append(dist[i][hard_neg_index].unsqueeze(0))

                if prob[hard_pos_index] >= threshold and prob[hard_neg_index] >= threshold:
                    # TP-TN
                    pass

                elif prob[hard_pos_index] >= threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # TP-FN
                    if is_FN:
                        tmp = rank_neg_index[1]
                        hard_neg_index_new = neg_idx[tmp]
                        j = 1
                        loop_cnt = 0
                        while prob[hard_neg_index_new] < threshold:
                            j += 1
                            tmp = rank_neg_index[j]
                            hard_neg_index_new = neg_idx[tmp]
                            loop_cnt += 1
                            if loop_cnt >= 10:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_ap[cnt] = weighty_new(dist[i][hard_pos_index].unsqueeze(0), dist[i][hard_neg_index].unsqueeze(0), pair_type=1, op_type=self.op_type)
                        dist_an[cnt] = dist[i][hard_neg_index_new].unsqueeze(0)
                    # TP-TN
                    else:
                        pass

                elif prob[hard_pos_index] < threshold and prob[hard_neg_index] >= threshold:
                    # FP-TN
                    tmp = rank_pos_index[-2]
                    hard_pos_index_new = pos_idx[tmp]
                    j = 2
                    loop_cnt = 0
                    while prob[hard_pos_index_new] < threshold:
                        j += 1
                        tmp = rank_pos_index[-j]
                        hard_pos_index_new = pos_idx[tmp]
                        loop_cnt += 1
                        if loop_cnt >= 5:
                            # print("------------warning, break the death loop---------------")
                            break
                    dist_an[cnt] = weighty_new(dist[i][hard_pos_index].unsqueeze(0), dist[i][hard_neg_index].unsqueeze(0), pair_type=0, op_type=self.op_type)
                    dist_ap[cnt] = dist[i][hard_pos_index_new].unsqueeze(0)

                elif prob[hard_pos_index] < threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # FP-FN
                    if is_FN:
                        loss_inverse = True
                    # FP-TN
                    else:
                        tmp = rank_pos_index[-2]
                        hard_pos_index_new = pos_idx[tmp]
                        j = 2
                        loop_cnt = 0
                        while prob[hard_pos_index_new] < threshold:
                            j += 1
                            tmp = rank_pos_index[-j]
                            hard_pos_index_new = pos_idx[tmp]
                            loop_cnt += 1
                            if loop_cnt >= 5:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_an[cnt] = weighty_new(dist[i][hard_pos_index].unsqueeze(0), dist[i][hard_neg_index].unsqueeze(0), pair_type=0, op_type=self.op_type)
                        dist_ap[cnt] = dist[i][hard_pos_index_new].unsqueeze(0)

                if loss_inverse:
                    loss += torch.clamp(dist_an[cnt] - dist_ap[cnt] + self.margin, 0)
                else:
                    loss += torch.clamp(dist_ap[cnt] - dist_an[cnt] + self.margin, 0)

                cnt += 1
                loss_inverse = False
            else:
                continue

        # compute accuracy
        if cnt == 0:
            return torch.Tensor([0.]).to(inputs.device), 0, cnt
        else:
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            correct = torch.ge(dist_an, dist_ap).sum().item()
            return loss / cnt, correct, cnt
        
def weighty_new(d1, d2, pair_type, op_type):
    if op_type == 'mean':
        return (d1 + d2) / 2.
    elif op_type == 'max':
        return max(d1, d2)
    elif op_type == 'min':
        return min(d1, d2)
    elif op_type == 'max-min':
        if pair_type == 1:  # pos
            return max(d1, d2)
        elif pair_type == 0:  # neg
            return min(d1, d2)
    elif op_type == 'weighty':
        if pair_type == 1:  # pos
            exp_sum = torch.exp(d1) + torch.exp(d2)
            d_sum = (torch.exp(d1) * d1 + torch.exp(d2) * d2) / exp_sum
        elif pair_type == 0:  # neg
            exp_sum = torch.exp(-d1) + torch.exp(-d2) + 1e-6  # avoid division by zero
            d_sum = (torch.exp(-d1) * d1 + torch.exp(-d2) * d2) / exp_sum
        return d_sum