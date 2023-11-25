import torch
import torch.nn as nn
from typing import List

class MultiPrototypeLoss(nn.Module):
    def __init__(self, num_prototypes_per_class: int):
        super(MultiPrototypeLoss, self).__init__()
        self.num_prototypes_per_class = num_prototypes_per_class

    def forward(self, batch_distances: torch.Tensor, labels: torch.Tensor):
        assert batch_distances.dim() == 2, "distances is expected to be of shape (batch_size, num_prototypes)"
        assert labels.dim() == 1, "labels is expected to be of shape (batch_size,)"
        batch_size = batch_distances.shape[0]
        
        batch_distances = batch_distances.view(batch_distances.size(0), batch_distances.size(1) // self.num_prototypes_per_class, self.num_prototypes_per_class)
        batch_distances = -torch.logsumexp(-batch_distances, dim=2)

        loss = 0
        for distances, label in zip(batch_distances, labels):
            in_class_distance = distances[label]
            out_class_distances = torch.cat((distances[:label], distances[label+1:]))
            # out = -torch.logsumexp(-out_class_distances, dim=0) # log sum exp (soft min)
            out = torch.mean(out_class_distances, dim=0)
            loss += in_class_distance.pow(2) / out.pow(2)
        loss /= batch_size
        return loss
