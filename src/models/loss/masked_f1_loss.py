import torch
import torch.nn as nn

class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        
        valid_mask = ~torch.isnan(target)
        valid_pred = pred * valid_mask
        valid_target = torch.where(valid_mask, target, torch.tensor(0.0, dtype=target.dtype))

        intersection = (valid_pred * valid_target).sum(dim=(-1, -2, -3))  # (TP)

        pred_sum = valid_pred.sum(dim=(-1, -2, -3))  # (TP + FP)
        target_sum = valid_target.sum(dim=(-1, -2, -3))  # (TP + FN)

        # F1 score
        f1 = (2 * intersection + self.epsilon) / (pred_sum + target_sum + self.epsilon)
        # return f1 loss
        return 1 - f1.mean()
