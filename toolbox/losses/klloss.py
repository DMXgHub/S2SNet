import torch.nn as nn
import torch.nn.functional as F
# from toolbox.loss.lovasz_loss import LovaszSoftmax
import torch


class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                        ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


# class MscLovaszSoftmaxLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
#         super(MscLovaszSoftmaxLoss, self).__init__()
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         if not isinstance(input, tuple):
#             input = (input,)
#
#         loss = 0
#         for item in input:
#             h, w = item.size(2), item.size(3)
#             item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
#             loss += LovaszSoftmax(item, item_target.squeeze(1).long())
#         return loss / len(input)


class FocalLossbyothers(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, weight=None, ignore_index=-100):
        super(FocalLossbyothers, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1-pt) ** self.gamma) * self.alpha * logpt
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, preds, labels):
        labels = labels
        preds = preds
        intersection = (preds * labels).sum(axis=(2, 3))
        unior = (preds + labels).sum(axis=(2, 3))
        dice = (2 * intersection + 1) / (unior + 1)
        dice = torch.mean(1 - dice)
        return dice


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.):
    """
    OFA-KD loss for semantic segmentation with BCHW format

    Args:
        logits_student: Student model logits of shape [B, C, H, W]
        logits_teacher: Teacher model logits of shape [B, C, H, W]
        target_mask: Ground truth mask of shape [B, C, H, W] or [B, 1, H, W]
        eps: Enhancement factor
        temperature: Temperature for softmax

    Returns:
        Loss value
    """
    # Apply softmax along the class dimension (dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # Apply target enhancement
    prod = (pred_teacher + target_mask) ** eps

    # Compute loss - need to handle spatial dimensions
    # We sum over classes (dim=1) and then average over spatial dimensions and batch
    log_pred_student = torch.log(pred_student + 1e-10)  # Add small epsilon for numerical stability

    # Sum over the class dimension
    loss = torch.sum(- (prod - target_mask) * log_pred_student, dim=1)

    # Average over spatial dimensions and batch
    loss = loss.mean()

    return loss
class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau



        self.KLD = torch.nn.KLDivLoss(reduction='sum')


    def forward(self, x_student, x_teacher):
        x_student = F.log_softmax(x_student / self.tau, dim=1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=1)
        # print(x_student.shape)
        b,c,h,w = x_student.shape
        numpix = b*h*w
        loss = self.KLD(x_student, x_teacher) / numpix
        # print("self.alpha", self.alpha)
        loss = self.alpha * loss
        return loss


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self, stu_map, tea_map):
        similiar = 1 - F.cosine_similarity(stu_map, tea_map, dim=1)
        loss = similiar.sum()
        return loss

if __name__ == '__main__':
    kld = KLDLoss()
    x = torch.randn(4, 4, 256, 256)
    y = torch.randn(4, 4, 256, 256)
    print(kld(x, y))