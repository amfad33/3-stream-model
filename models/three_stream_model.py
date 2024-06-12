import torch
import torch.nn as nn
import torch.nn.functional as F
import models.spatial_stream as s_s
import models.temporal_stream as t_s
import models.audio_stream as a_s
from torchsummary import summary


class MixtureOfExperts(nn.Module):
    def __init__(self, c):
        super(MixtureOfExperts, self).__init__()
        # Initialize the weights as a single tensor
        self.g = nn.Parameter(torch.rand(3, c))
        nn.init.xavier_uniform_(self.g)

    def forward(self, V_S, V_T, V_A):
        # Apply softmax to the weights along the first dimension
        g_softmax = F.softmax(self.g, dim=0)

        # Split the softmax results into g_S, g_T, g_A
        g_S, g_T, g_A = g_softmax[0], g_softmax[1], g_softmax[2]

        # Compute the weighted sum of the input tensors
        return V_S * g_S + V_T * g_T + V_A * g_A


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


class ThreeStreamModel(nn.Module):
    def __init__(self, c, audio_input_size, audio_frame_count):
        super(ThreeStreamModel, self).__init__()
        self.stream1 = s_s.ResNet(c)
        self.stream2 = t_s.SlowFusionVideoModelSharedWeightsFC(3, c)
        self.stream3 = a_s.FullyConnectedAudioNet(audio_input_size, c, audio_frame_count)
        self.moe = MixtureOfExperts(c)
        self.cg = ContextGating(c)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, frames, audio):
        V_S = self.stream1(image)
        V_T = self.stream2(frames)
        V_A = self.stream3(audio)
        output = self.moe(V_S, V_T, V_A)
        output = self.cg(output)
        output = self.softmax(output)
        return output


# Used if all streams are separately trained
class ThreeStreamModelTrainedStreams(nn.Module):
    def __init__(self, c, audio_input_size, audio_frame_count):
        super(ThreeStreamModelTrainedStreams, self).__init__()
        self.moe = MixtureOfExperts(c)
        self.cg = ContextGating(c)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, V_S, V_T, V_A):
        output = self.moe(V_S, V_T, V_A)
        output = self.cg(output)
        output = self.softmax(output)
        return output


# Example usage:
# model = ThreeStreamModel(4800, 2048, 10)
# summary(model.stream1, (3, 224, 224))
# summary(model.stream2, (3, 30, 224, 224))
# summary(model.stream3, (10, 2048))
