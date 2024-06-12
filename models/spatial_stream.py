import torch.nn as nn
import torchvision.models as models


# This is the spatial stream. It will receive the central frame as input and will create a vector of c values as output
class ResNet(nn.Module):
    def __init__(self, c):
        super(ResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, c)
        nn.init.xavier_uniform_(self.resnet.fc.weight)

    def forward(self, x):
        x = self.resnet(x)
        return x


# Example usage:
# model = ResNet101(num_classes=10)
# input_tensor = torch.randn(8, 3, 224, 224)  # Batch of 8 videos, each with 30 frames of 224x224 RGB images
# output = model(input_tensor)
# print(output.shape)  # Should be [8, 10]
