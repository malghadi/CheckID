import torch
import torch.nn as nn
import torch.nn.functional as F


class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        # Fully connected layer
        self.fc1 = torch.nn.Linear(2048, 1024)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(1024, 128)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(100, 32)        # convert matrix with 84 features to a matrix of 10 features (columns)
        self.fc4 = torch.nn.Linear(32, 2)
    
    def forward(self, x):
        # x = x.view(-1, 7*7*1664)
        # FC-1, then perform ReLU non-linearity
        # x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        # x = torch.nn.functional.relu(self.fc2(x))
        # FC-3 then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc3(x))
        # FC-4
        x = self.fc4(x)
        y1o_softmax = F.softmax(x, dim=1)
        return x, y1o_softmax


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
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

        self.classify = Classify()
        self.fc4 = torch.nn.Linear(5, 2)

    def forward(self, input):
        output = self.cnn1(input)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        # class_op_1, y1o_softmax = self.classify(output)
        class_op_1 = self.fc4(output)
        y1o_softmax = F.softmax(class_op_1, dim=1)
        return output, class_op_1, y1o_softmax

