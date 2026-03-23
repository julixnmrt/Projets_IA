import torch
import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):

    def __init__(self, channels=[3,32,64,128]):
        super().__init__()

        layers = []

        for i in range(len(channels)-1):
            layers.append(ConvBlock(channels[i], channels[i+1]))

        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(channels[-1], 256),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, 10)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x