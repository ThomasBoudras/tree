import torch
import torch.nn as nn

class NMAELoss(nn.Module):
    def __init__(self):
        super(NMAELoss, self).__init__()
    
    def forward(self, pred, target):
        absolute_error = torch.abs(pred - target)
        nmae = torch.mean(absolute_error / (1 + torch.abs(target)))
        return nmae
