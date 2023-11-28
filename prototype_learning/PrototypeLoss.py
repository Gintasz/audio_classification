import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class PrototypeLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(PrototypeLoss, self).__init__()
        self.num_classes = num_classes
        self.one_hot_classes = F.one_hot(torch.arange(0, num_classes), num_classes=num_classes)
    
    def get_one_hot_encoding_for_class(self, class_id: int):
        # class ids starts with 1
        return self.one_hot_classes[class_id - 1]
        
    gamma = 0.000001
    def forward(self, batch_distances: torch.Tensor, labels: torch.Tensor):
        assert batch_distances.dim() == 2, "distances is expected to be of shape (batch_size, num_prototypes)"
        assert labels.dim() == 1, "labels is expected to be of shape (batch_size,)"
        labels_one_hot_encoded = torch.stack([self.get_one_hot_encoding_for_class(class_id) for class_id in labels])
        
        batch_size = batch_distances.shape[0]
        loss = 0
        
        # tau = 0.5
        # batch_cluster_attributions = F.gumbel_softmax(-batch_distances, tau=tau, hard=False)
        
        # for distances_to_prototype, attributed_to_prototype in zip(batch_distances.t(), batch_cluster_attributions.t()):
        #     distances_to_prototype_after_attribution = distances_to_prototype * attributed_to_prototype
        #     column = distances_to_prototype_after_attribution
            
        #     # count number of attributions
        #     attribution_class_counts = (attributed_to_prototype.view(-1, 1) * labels_one_hot_encoded).sum(dim=0).squeeze(0)
        #     attribution_class_counts_argmax = F.gumbel_softmax(attribution_class_counts, tau=tau, hard=False)
            
            
        #     probabilities = attribution_class_counts / (attribution_class_counts.sum() + 1e-5)
        #     # entropy = -torch.sum(probabilities * torch.log(probabilities))
        #     # loss += entropy
        #     ipr = (probabilities.pow(4).sum()) / (probabilities.pow(2).sum().pow(2) + 1e-5)
        #     loss += -ipr
        #     continue
            
        #     attributed_to_prototype * labels
            
        #     column_one_hot = column.view(-1, 1) * labels_one_hot_encoded
            
        #     column_one_hot_summed = column_one_hot.sum(dim=0).squeeze(0)
            
        #     column_one_hot_summed_argmax = F.gumbel_softmax(column_one_hot_summed, tau=tau, hard=False)
            
        #     most_popular_class_sum = (column_one_hot_summed * column_one_hot_summed_argmax).sum()
        #     other_classes_sum = ((column_one_hot_summed * (column_one_hot_summed_argmax - 1)) * -1).sum()
        #     #most_popular_class_sum = (column_one_hot * column_one_hot_summed_argmax.view(1, -1)).sum(dim=1).sum(dim=0)
        #     #other_classes_sum = -1 * (column_one_hot * (column_one_hot_summed_argmax - 1).view(1, -1)).sum(dim=1).sum(dim=0)
            
        #     loss += most_popular_class_sum - other_classes_sum
        #     # loss += other_classes_sum / (most_popular_class_sum + 1e-5)
        #     # ipr = (column_one_hot_summed.pow(4).sum()) / (column_one_hot_summed.pow(2).sum().pow(2) + 1e-5)
        #     # loss += -ipr
            
        
        # old but working loss
        for distances, label in zip(batch_distances, labels):
            in_class_distance = distances[label]
            out_class_distances = torch.cat((distances[:label], distances[label + 1:]))
            out = -torch.logsumexp(-out_class_distances, dim=0) # log sum exp (soft min)
            #loss += in_class_distance.pow(2) / out.pow(2) # 0.95% accuracy on training
            # loss += in_class_distance / out
            loss += (in_class_distance - distances.min()) / (distances - distances.min()).sum()
        loss /= batch_size
        
        
        #loss += in_class_distance / distances.sum()
        
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Robust_Classification_With_CVPR_2018_paper.pdf
        num_prototypes_per_class = 1
        
        # for distances, label in zip(batch_distances, labels):
        #     scaled_distances = distances
        #     assert (-self.gamma * scaled_distances[label]) != 0
        #     assert not torch.isnan((-self.gamma * scaled_distances[label]))
        #     if (-self.gamma * scaled_distances).exp().sum() == 0:
        #         print(scaled_distances)
        #     assert (-self.gamma * scaled_distances).exp().sum() != 0
        #     assert not torch.isnan((-self.gamma * scaled_distances).exp().sum())
        #     assert (-self.gamma * scaled_distances[label])/(-self.gamma * scaled_distances).exp().sum() + 1e-5 > 0, "log of negative number"
        #     l = -torch.log((-self.gamma * scaled_distances[label])/((-self.gamma * scaled_distances).exp().sum() + 1e-5))
        #     assert l != 0 and not torch.isnan(l)
        #     loss += l
        #     # p_y_x = 0
        #     # for j in range(0, num_prototypes_per_class):
        #     #     p_x_mij = exp(-gamma * )
        #     #     p_y_x += p_x_mij
        #     # loss_datapoint = - log()
        return loss
