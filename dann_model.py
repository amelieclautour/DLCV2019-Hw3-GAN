import torch.nn as nn
from functions import ReverseLayerF


class DANN_Neural_Network(nn.Module):

    def __init__(self):
        super(DANN_Neural_Network, self).__init__()
        self.attr = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True))

        self.cclass = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1))

        self.domain = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1))

    def forward(self, input, cst):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        attr = self.attr(input)
        attr = attr.view(-1, 50 * 4 * 4)
        reverse_attr = ReverseLayerF.apply(attr, cst)
        class_output = self.cclass(attr)
        domain_output = self.domain(reverse_attr)

        # return class_output, domain_output
        return class_output, domain_output, attr