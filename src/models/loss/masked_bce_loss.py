import torch
import torch.nn as nn
import torch.nn.functional as F

class masked_bce_loss(nn.Module):
    def __init__(self, reduction='mean'):  # La BCE_loss n'a pas de paramètre beta
        super(masked_bce_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        mask = ~torch.isnan(target)  # Crée un masque où target n'est pas NaN
        return F.binary_cross_entropy(pred[mask], target[mask].float(), reduction=self.reduction)
        