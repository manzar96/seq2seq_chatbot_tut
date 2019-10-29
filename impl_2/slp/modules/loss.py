import torch
import torch.nn as nn


class RMSELoss(nn.Module):

    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # used to avoid nan values if mse loss is zero.

    def forward(self, prediction, target):
        return torch.sqrt(self.mse(prediction, target) + self.eps)


