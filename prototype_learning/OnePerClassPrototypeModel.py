import torch
import torch.nn as nn
from typing import List
import torch
from .PrototypeRepresentationLayer import PrototypeRepresentationLayer
from .PrototypeLoss import PrototypeLoss

class OnePerClassPrototypeModel(nn.Module):
    def __init__(self):
        super(OnePerClassPrototypeModel, self).__init__()
        self.prototype_layer = PrototypeRepresentationLayer(num_prototypes=4, num_timesteps=81, num_coeffs=13 * 3)
        self.loss_function = PrototypeLoss(num_classes=4)
        
        self.sub_params = nn.Parameter(torch.randn(39))
        self.mult_params = nn.Parameter(torch.randn(39))

    def forward(self, x: torch.Tensor):
        x = x - self.sub_params.unsqueeze(0).unsqueeze(0)
        x = x * self.mult_params.unsqueeze(0).unsqueeze(0)

        y = self.prototype_layer.forward(x)
        y_argmin = torch.argmin(y, dim=1)
        return y, y_argmin
        
    def loss(self, out, labels: torch.Tensor):
        y, y_argmin = out
        return self.loss_function(y, labels)
