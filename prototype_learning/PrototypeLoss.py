import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class PrototypeLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(PrototypeLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, batch_distances: torch.Tensor, labels: torch.Tensor):
        assert batch_distances.dim() == 2, "distances is expected to be of shape (batch_size, num_prototypes)"
        assert labels.dim() == 1, "labels is expected to be of shape (batch_size,)"
        
        batch_size = batch_distances.shape[0]
        loss = torch.tensor(0.0)
        
        for distances, label in zip(batch_distances, labels):
            correct_class_distance = distances[label]
            incorrect_class_distances = torch.cat((distances[:label], distances[label + 1:]))
            min_dist_incorrect = -torch.logsumexp(-incorrect_class_distances, dim=0) # log sum exp (soft min)
            #loss += in_class_distance.pow(2) / min_dist_incorrect.pow(2) # 0.95% accuracy on training
            # loss += (in_class_distance - distances.min()) / (distances - distances.min()).sum() # 98.5% accuracy on training with augment
            # loss += correct_class_distance / min_dist_incorrect
            # loss += (in_class_distance - distances.min()) / (distances).sum()
            # if torch.argmin(distances) != label:
            loss += correct_class_distance / min_dist_incorrect
            # else:
                # pass
        loss /= batch_size
        return loss
