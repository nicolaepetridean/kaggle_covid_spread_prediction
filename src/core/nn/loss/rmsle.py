import torch
from torch import nn
import numpy as np

# =============================================== RMSLE ================================================================
class RMSLELoss(nn.Module):
    def __init__(self):
        '''
        Root Mean Squared Log Error
        '''
        super().__init__()

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor, target):
        error = torch.log(inTensor + 1 + np.spacing(1)) - torch.log(target + 1 + np.spacing(1))
        return torch.sqrt(torch.pow(error, 2).mean())