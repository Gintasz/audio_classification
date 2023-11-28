import torch
import torch.nn as nn
import pysdtw
from typing import List
import torch

class PrototypeRepresentationLayer(nn.Module):
    def __init__(self, num_prototypes: int, num_timesteps: int, num_coeffs: int):
        super(PrototypeRepresentationLayer, self).__init__()
        self.num_prototypes = num_prototypes

        # Initialize the prototype MFCC vectors as parameters
        self.prototypes = nn.ParameterList([nn.Parameter(torch.rand(num_timesteps, num_coeffs), requires_grad=True) for i in range(num_prototypes)])
        fun = pysdtw.distance.pairwise_l2_squared
        self.sdtw = pysdtw.SoftDTW(gamma=0.1, dist_func=fun, use_cuda=torch.cuda.is_available())

    def forward(self, x: torch.Tensor):
        # Compute DTW distance between input x and all prototypes
        assert x.dim() == 3, "x is expected to be of shape (batch_size, num_timesteps, n_mfcc)"
        batch_size = x.shape[0]
        prototypes_tensor = torch.stack([p for p in self.prototypes], dim=0)
        x_b = x.repeat_interleave(self.num_prototypes, dim=0)
        y_b = prototypes_tensor.repeat(batch_size, 1, 1)
        
        distances = self.sdtw(x_b, y_b)
        distances_reshaped = distances.view(-1, self.num_prototypes)
        assert torch.isfinite(distances_reshaped).all(), "no distance element shall be nan"
        return distances_reshaped
    
