import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        bce = F.binary_cross_entropy_with_logits(x, y)
        smooth = 1e-5
        x = torch.sigmoid(x)
        num = y.size(0)
        x = x.view(num, -1)
        y = y.view(num, -1)
        intersection = (x * y)
        dice = (2. * intersection.sum(1) + smooth) / (x.sum(1) + y.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.squeeze(1)
        y = y.squeeze(1)
        loss = lovasz_hinge(x, y, per_image=True)

        return loss
