import torch
from torch.nn import functional as F


def calc_loc_loss(pred_loc: torch.Tensor,
                  test_loc: torch.Tensor,
                  cls_label: torch.Tensor,
                  use_smooth: bool = True
                  ):# -> torch.Tensor:
    """Compute location regression loss only on positive samples.

    :param pred_loc: Predicted bbox offsets. Sized [N, S, 2].
    :param test_loc: Ground truth bbox offsets. Sized [N, S, 2].
    :param cls_label: Class labels where the 1 marks the positive samples. Sized
        [N, S].
    :param use_smooth: If true, use smooth L1 loss. Otherwise, use L1 loss.
    :return: Scalar loss value.
    """
    pos_idx = cls_label.eq(1).unsqueeze(-1).repeat((1, 1, 2))

    pred_loc = pred_loc[pos_idx]
    test_loc = test_loc[pos_idx]

    if use_smooth:
        loc_loss = F.smooth_l1_loss(pred_loc, test_loc)
    else:
        loc_loss = (pred_loc - test_loc).abs().mean()

    return loc_loss


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form.

    :param labels: Class labels. Sized [N].
    :param num_classes: Number of classes.
    :return: One-hot encoded labels. sized [N, #classes].
    """
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]

def focal_loss(x: torch.Tensor,
               y: torch.Tensor,
               alpha: float = 0.25,
            #    alpha: float = 0.1,
            #    alpha: float = 0.1,
               gamma: float = 2,
               reduction: str = 'sum'
               ) -> torch.Tensor:
    """Compute focal loss for binary classification.
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    :param x: Predicted confidence. Sized [N, D].
    :param y: Ground truth label. Sized [N].
    :param alpha: Alpha parameter in focal loss.
    :param gamma: Gamma parameter in focal loss.
    :param reduction: Aggregation type. Choose from (sum, mean, none).
    :return: Scalar loss value.
    """
    _, num_classes = x.shape

    t = one_hot_embedding(y, num_classes)
    esp = 1e-8
    # esp = 0
    # p_t = p if t > 0 else 1-p
    p_t = x * t + (1 - x) * (1 - t)+esp
    # alpha_t = alpha if t > 0 else 1-alpha
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()

    if reduction == 'sum':
        fl = fl.sum()
    elif reduction == 'mean':
        fl = fl.mean()
    elif reduction == 'none':
        pass
    else:
        # raise ValueError(f'Invalid reduction mode {reduction}')
        raise ValueError('Invalid reduction mode {reduction}')

    return fl


def focal_loss_with_logits(x, y, reduction='sum'):
    """Compute focal loss with logits input"""
    return focal_loss(x.sigmoid(), y, reduction=reduction)

def calc_ctr_loss(pred, test, pos_mask):
    pos_mask = pos_mask.type(torch.bool)

    pred = pred[pos_mask]
    test = test[pos_mask]

    loss = F.binary_cross_entropy(pred, test)
    return loss



def calc_cls_a_loss(pred: torch.Tensor,
                  test: torch.Tensor,
                  kind: str = 'focal'
                  ) -> torch.Tensor:
    """Compute classification loss on both positive and negative samples.

    :param pred: Predicted class. Sized [N, S].
    :param test: Class label where 1 marks positive, -1 marks negative, and 0
        marks ignored. Sized [N, S].
    :param kind: Loss type. Choose from (focal, cross-entropy).
    :return: Scalar loss value.
    """
    test = test.type(torch.long)
    num_pos = test.sum()
    esp = 1e-8
    # esp = 0
    pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)+esp

    if kind == 'focal':
        loss = focal_loss(pred, test, reduction='sum')
    elif kind == 'cross-entropy':
        loss = F.nll_loss(pred.log(), test)
    else:
        # raise ValueError(f'Invalid loss type {kind}')
        raise ValueError('Invalid loss type {kind}')

    loss = loss / num_pos
    return loss



def calc_cls_loss(pred: torch.Tensor, test: torch.Tensor):#  -> torch.Tensor:
    """Compute classification loss.

    :param pred: Predicted confidence (0-1). Sized [N, S].
    :param test: Class label where 1 marks positive, -1 marks negative, and 0
        marks ignored. Sized [N, S].
    :return: Scalar loss value.
    """
    pred = pred.view(-1)
    test = test.view(-1)
    esp = 1e-8
    # esp = 0
    pos_idx = test.eq(1).nonzero().squeeze(-1)
    pred_pos = pred[pos_idx].unsqueeze(-1)+esp
    pred_pos = torch.cat([1 - pred_pos, pred_pos], dim=-1)+esp
    gt_pos = torch.ones(pred_pos.shape[0], dtype=torch.long, device=pred.device)
    loss_pos = F.nll_loss(pred_pos.log(), gt_pos)

    neg_idx = test.eq(-1).nonzero().squeeze(-1)
    pred_neg = pred[neg_idx].unsqueeze(-1)
    pred_neg = torch.cat([1 - pred_neg, pred_neg], dim=-1)+esp
    gt_neg = torch.zeros(pred_neg.shape[0], dtype=torch.long,
                         device=pred.device)
    loss_neg = F.nll_loss(pred_neg.log(), gt_neg)

    loss = (loss_pos + loss_neg) * 0.5
    return loss
