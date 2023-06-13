import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, n_actions, n_conditions):
        super().__init__()
        self.model = models.resnet18(weights='DEFAULT')

        in_features = self.model.fc.in_features
        self.fc = nn.Linear(in_features + n_conditions, n_actions)

        self.model = nn.Sequential(*(list(self.model.children())[:-1]))


    def forward(self, x, condition):
        out = self.model(x)
        # print(f"{out.flatten().shape = }")
        # print(f"{condition.shape = }")
        out = torch.cat((out.flatten(), condition.flatten()))
        out = self.fc(out)
        return out
