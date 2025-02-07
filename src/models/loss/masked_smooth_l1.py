import torch
import torch.nn.functional as F
import torch.nn as nn

class masked_smooth_l1(nn.Module):
    def __init__(self, reduction, beta):
        super(masked_smooth_l1, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target):
        mask = ~torch.isnan(target)  # Crée un masque où target n'est pas NaN
        return F.smooth_l1_loss(pred[mask], target[mask], reduction=self.reduction, beta=self.beta)
