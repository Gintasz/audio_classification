import torch
import torch.nn as nn
from typing import List
import torch
from .PrototypeRepresentationLayer import PrototypeRepresentationLayer
from .PrototypeLoss import PrototypeLoss
import torch.nn.functional as F

class OnePerClassPrototypeModel2(nn.Module):
    def __init__(self):
        super(OnePerClassPrototypeModel2, self).__init__()
        self.prototype_layer = PrototypeRepresentationLayer(num_prototypes=4, num_timesteps=32, num_coeffs=13 * 3,)
        self.loss_function = PrototypeLoss(num_classes=4)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding='same')
        self.conv2 = nn.Conv2d(32, 16, kernel_size=4, stride=1, padding='same')
        self.conv3 = nn.Conv2d(16, 8, kernel_size=4, stride=1, padding='same')
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(1)
        
        # self.fc1 = nn.Linear(64 * 4 * 13, 32 * 13 * 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = (F.relu(self.bn1(self.conv1(x))))
        x = (F.relu(self.bn2(self.conv2(x))))
        x = (F.relu(self.bn3(self.conv3(x))))
        x = (F.relu(self.bn4(self.conv4(x))))
        x = x.squeeze(1)
        
        y = self.prototype_layer.forward(x)
        y_argmin = torch.argmin(y, dim=1)
        return y, y_argmin
        
    def loss(self, out, labels: torch.Tensor):
        y, y_argmin = out
        return self.loss_function(y, labels)