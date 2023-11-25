import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_prototypes: int):
        super(CrossEntropyLoss, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_prototypes, affine=False)

    def forward(self, batch_distances: torch.Tensor, labels: torch.Tensor):
        assert batch_distances.dim() == 2, "distances is expected to be of shape (batch_size, num_prototypes)"
        assert labels.dim() == 1, "labels is expected to be of shape (batch_size,)"
        batch_size = batch_distances.shape[0]
        num_prototypes = batch_distances.shape[1]
        
        # Log-Sum-Exp trick
        max_distance = torch.max(batch_distances, dim=1, keepdim=True)[0]
        mean_distance = torch.mean(batch_distances, dim=1, keepdim=True)[0]
        stabilized_distances = batch_distances - mean_distance
        normalized_distances = self.batch_norm(batch_distances)
        
        negative_batch_distances = -stabilized_distances
        probabilities = torch.nn.functional.softmax(negative_batch_distances, dim=1)
        
        target_distribution = one_hot(labels, num_classes=num_prototypes).float()
        
        loss = -torch.sum(target_distribution * torch.log(probabilities + 1e-9), dim=1)
        loss = loss.mean()
        return loss
