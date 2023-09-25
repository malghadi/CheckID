import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 240 * 240, 100), # pow(2,18)
            nn.ReLU(inplace=True),

            nn.Linear(100, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 5)
            )

    def forward(self, input):
        output = self.cnn1(input)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Fully connected layer
        self.fc4 = nn.Linear(32, 2)
    
    def forward(self, x):
        # FC-4
        x = self.fc4(x)
        y1o_softmax = F.softmax(x, dim=1)
        return x, y1o_softmax
