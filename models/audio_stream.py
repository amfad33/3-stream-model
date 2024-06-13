import torch.nn as nn
import torch.nn.functional as F
import models.nextvlad as nv


class FullyConnectedAudioNet(nn.Module):
    def __init__(self, input_size, c, frame_count):
        super(FullyConnectedAudioNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.nextvlad = nv.NeXtVLAD(dim=512, num_clusters=32, lamb=2, groups=16, max_frames=frame_count)
        self.fc3 = nn.Linear(2048, c)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.nextvlad(x)
        x = F.relu(self.fc3(x))
        return x

# Example usage:
# model = FullyConnectedAudioNet(128, 10, 30)
# input_tensor = torch.randn(8, 30, 128)  # Batch of 8 videos, each with 30 frames of 128 audio features
# output = model(input_tensor)
# print(output.shape)  # Should be [8, 10]
