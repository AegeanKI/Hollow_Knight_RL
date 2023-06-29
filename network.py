import torch
import torch.nn as nn

from torchvision import models

class ResNetLSTM(nn.Module):
    def __init__(self, out_classes, in_condition_size,
                 lstm_layer=1, lstm_hidden_size=512):
        super().__init__()
        self.encoder = models.resnet34(weights='DEFAULT')
        self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1])) # output size: 2048

        self.lstm_layer = lstm_layer
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(512 + in_condition_size, self.lstm_hidden_size, self.lstm_layer)
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_classes)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, condition):
        out = self.encoder(x)

        self.h = torch.zeros((self.lstm_layer, self.lstm_hidden_size), dtype=torch.float32).to(self.device)
        self.c = torch.zeros((self.lstm_layer, self.lstm_hidden_size), dtype=torch.float32).to(self.device)
        out = out.view(out.size(0), out.size(1))
        out, _ = self.lstm(torch.cat((out, condition), dim=1), (self.h, self.c))

        out = self.decoder(out)
        return out


# class ResNet18(nn.Module):
#     def __init__(self, n_actions, in_condition_size):
#         super().__init__()
#         # self.model = models.resnet18(weights='DEFAULT')
#         self.model = ResNet(img_channels=n_frames, num_layers=18,
#                             in_condition_size=n_conditions, num_classes=n_actions)

#     @property
#     def device(self):
#         return next(model.parameters()).device

#     def forward(self, x, condition):
#         out = self.model(x, condition)
#         return out
