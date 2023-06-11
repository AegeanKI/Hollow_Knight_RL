import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.model = models.resnet18(weights='DEFAULT')

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.model(x)
        return out
