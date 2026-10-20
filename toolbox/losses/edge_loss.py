import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def logit(x):
#     eps = 1e-7
#     x = torch.clamp(x,eps,1-eps)
#     x = torch.log(x / (1 - x))
#     return x

def cross_entropy(logits,labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

def weighted_cross_entropy(logits,labels,alpha):
    return torch.mean((1 - alpha) * ((1 - labels) * logits + torch.log(1 + torch.exp(-logits))) + (2 * alpha - 1) * labels * torch.log(1 + torch.exp(-logits)))

class EdgeHoldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace = torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3])
        #filter shape in Pytorch: out_channel, in_channel, height, width
        self.laplace = nn.Parameter(data=laplace,requires_grad=False)
    def torchLaplace(self,x):
        edge = F.conv2d(x.float(),self.laplace.cuda(),padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge
    def forward(self,y_pred,y_true):
        y_pred = nn.Sigmoid()(y_pred)
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = cross_entropy(y_pred_edge,y_true_edge)
        saliency_loss = weighted_cross_entropy(y_pred,y_true,alpha=0.528)
        # if mode == 'debug':
        #     print('edge loss:',edge_loss.item(),'saliency loss:',saliency_loss.item())
        return 0.8 * saliency_loss + 0.2 * edge_loss
        # return  edge_loss
