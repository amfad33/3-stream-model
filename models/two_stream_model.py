import torch
import torch.nn as nn
import torch.nn.functional as F
import models.spatial_stream as s_s
import models.temporal_stream as t_s
import models.MoE as MoE
from torchsummary import summary


class MixtureOfExperts(nn.Module):
    def __init__(self, c):
        super(MixtureOfExperts, self).__init__()
        # Initialize the weights as a single tensor
        self.g = nn.Parameter(torch.rand(2, c))
        nn.init.xavier_uniform_(self.g)

    def forward(self, V_S, V_T):
        # Apply softmax to the weights along the first dimension
        g = F.softmax(self.g, dim=0)

        # Split the softmax results into g_S, g_T
        g_S, g_T = g[0], g[1]

        # Compute the weighted sum of the input tensors
        return V_S * g_S + V_T * g_T


class ContextGating(nn.Module):
    def __init__(self, c):
        super(ContextGating, self).__init__()
        self.W = nn.Parameter(torch.rand(c, c))
        self.b = nn.Parameter(torch.rand(c))
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x):
        gate_values = torch.sigmoid(F.linear(x, self.W, self.b))
        return gate_values * x


class TwoStreamModel(nn.Module):
    def __init__(self, c):
        super(TwoStreamModel, self).__init__()
        self.stream1 = s_s.ResNet(c)
        self.stream2 = t_s.SlowFusionVideoModelSharedWeightsFC(3, c)
        # self.moe = MixtureOfExperts(c)
        # self.cg = ContextGating(c)
        self.moe = MoE.MoeModel(c, c,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, frames):
        V_S = self.stream1(image)
        V_T = self.stream2(frames)
        output = self.moe(torch.stack([V_S, V_T], dim=1))
        # output = self.cg(output)
        output = self.softmax(output)
        return output


# This model is used for when streams are trained seprately and you want to train the MoE weights and their
# augmentations
class TwoStreamModelTrainedStreams(nn.Module):
    def __init__(self, c):
        super(TwoStreamModelTrainedStreams, self).__init__()
        # self.moe = MixtureOfExperts(c)
        # self.cg = ContextGating(c)
        self.moe = MoE.MoeModel(c, c, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, V_S, V_T):
        output = self.moe(torch.stack([V_S, V_T], dim=1))
        # output = self.cg(output)
        output = self.softmax(output)
        return output


# Example usage:
# model = TwoStreamModel(4800)
# summary(model.stream1, (3, 224, 224))
# summary(model.stream2, (3, 30, 224, 224))
