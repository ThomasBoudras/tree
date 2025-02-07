import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMaskedBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedMaskedBCELoss, self).__init__()
        self.reduction = reduction  # 'mean' ou 'sum' pour la normalisation

    def forward(self, pred, target):
        # Créer un masque pour exclure les NaN
        mask = ~torch.isnan(target)
        pred, target = pred[mask], target[mask].float()
        
        # Comptage des pixels positifs et négatifs après application du masque
        num_pos = target.sum()  # Nombre de pixels True (1)
        num_neg = target.numel() - num_pos  # Nombre de pixels False (0)

        # Éviter la division par zéro
        num_pos = num_pos.clamp(min=1.0)
        num_neg = num_neg.clamp(min=1.0)

        # Calcul des poids (inversement proportionnels)
        weight_pos = num_neg / (num_pos + num_neg)  # Poids des pixels True
        weight_neg = num_pos / (num_pos + num_neg)  # Poids des pixels False

        # Application des poids à la BCE
        loss = - (weight_pos * target * torch.log(pred + 1e-8) + 
                  weight_neg * (1 - target) * torch.log(1 - pred + 1e-8))

        # Réduction (mean ou sum)
        return loss.mean() if self.reduction == 'mean' else loss.sum()
