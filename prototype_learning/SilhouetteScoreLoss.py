import torch
import torch.nn as nn
from typing import List

class SilhouetteScoreLoss(nn.Module):
    def __init__(self):
        super(SilhouetteScoreLoss, self).__init__()

    def forward(self, batch_distances: torch.Tensor, labels: torch.Tensor):
        assert batch_distances.dim() == 2, "distances is expected to be of shape (batch_size, num_prototypes)"
        assert labels.dim() == 1, "labels is expected to be of shape (batch_size,)"
        batch_size, num_prototypes = batch_distances.size()
        
        # Create a mask for distances to the own prototype
        mask = torch.nn.functional.one_hot(labels, num_prototypes).bool()
        
        # Calculate a: Mean intra-cluster distance
        a = batch_distances.masked_select(mask).view(batch_size, -1).mean(1)
        
        # Calculate b: Mean nearest-cluster distance
        # We use ~mask to exclude the own cluster distances
        b = batch_distances.masked_select(~mask).view(batch_size, -1).min(1)[0]
        
        # Apply the log-sum-exp trick for max(a, b)
        max_ab = torch.logsumexp(torch.stack((a, b), dim=1), dim=1)
        
        # Calculate the silhouette scores
        silhouette_scores = (b - a) / max_ab
        
        # Since we want to minimize the loss, and a higher silhouette score is better,
        # we can minimize the negative silhouette score.
        loss = -silhouette_scores.mean()  # Averaging over the batch
        
        return loss