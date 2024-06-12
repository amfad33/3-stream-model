import torch
import torch.nn as nn
import torch.nn.functional as F
import models.nextvlad as nv


# Define the slow fusion video model with nextvlad and a single fully connected layer
class SlowFusionVideoModelSharedWeightsFC(nn.Module):
    def __init__(self, in_channels, c):
        super(SlowFusionVideoModelSharedWeightsFC, self).__init__()

        # First layer:
        self.conv1 = nn.ModuleList([
            nn.Conv3d(in_channels, 64, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
            for _ in range(14)])
        self.bn1 = nn.ModuleList([
            nn.BatchNorm3d(64)
            for _ in range(14)])
        self.pool1 = nn.ModuleList([
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
            for _ in range(14)])

        # Second layer:
        self.conv2 = nn.ModuleList([
            nn.Conv3d(64, 128, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
            for _ in range(6)])
        self.bn2 = nn.ModuleList([
            nn.BatchNorm3d(128)
            for _ in range(6)])
        self.pool2 = nn.ModuleList([
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
            for _ in range(6)])

        # Third layer:
        self.conv3 = nn.ModuleList([
            nn.Conv3d(128, 256, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
            for _ in range(3)])
        self.bn3 = nn.ModuleList([
            nn.BatchNorm3d(256)
            for _ in range(3)])
        self.pool3 = nn.ModuleList([
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
            for _ in range(3)])

        # Fourth layer:
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        # Fully connected and NextVLad
        self.nextvlad = nv.NeXtVLAD(dim=196, num_clusters=64, lamb=2, groups=8, max_frames=512)
        self.fc = nn.Linear(3136, c)

    def forward(self, x):
        # First layer:
        out1 = []
        for i in range(14):
            idx_start = i * 2
            idx_end = idx_start + 4
            frames = x[:, :, idx_start:idx_end, :, :]
            out = self.conv1[i](frames)
            out = self.bn1[i](out)
            out = self.pool1[i](out)
            out1.append(out)
        out1 = torch.cat(out1, dim=2)

        out2 = []
        for i in range(6):
            idx_start = i * 2
            idx_end = idx_start + 4
            frames = out1[:, :, idx_start:idx_end, :, :]
            out = self.conv2[i](frames)
            out = self.bn2[i](out)
            out = self.pool2[i](out)
            out2.append(out)
        out2 = torch.cat(out2, dim=2)

        out3 = []
        for i in range(3):
            idx_start = i * 2
            idx_end = idx_start + 2
            frames = out2[:, :, idx_start:idx_end, :, :]
            out = self.conv3[i](frames)
            out = self.bn3[i](out)
            out = self.pool3[i](out)
            out3.append(out)
        out3 = torch.cat(out3, dim=2)

        out4 = self.conv4(out3)
        out4 = self.bn4(out4)
        out4 = self.pool4(out4)

        out4 = out4.squeeze(2)
        out4 = out4.view(out4.shape[0], 512, 196)

        out = self.nextvlad(out4)
        out = F.relu(self.fc(out))
        return out


# Example usage:
# model = SlowFusionVideoModel(num_classes=10)
# input_tensor = torch.randn(8, 3, 30, 224, 224)  # Batch of 8 videos, each with 30 frames of 224x224 RGB images
# output = model(input_tensor)
# print(output.shape)  # Should be [8, 10]
