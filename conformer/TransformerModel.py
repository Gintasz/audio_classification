import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

class TransformerModel(nn.Module):
    def __init__(self, num_classes: int):
        super(TransformerModel, self).__init__()
        self.num_classes = num_classes
        
        self.transformer1 = Transformer(
            d_model=128, # number of coeffs per single timestep
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            batch_first=True
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        lengths = torch.full((batch_size,), 81, dtype=torch.int64)
        lengths = lengths.to(x.device)
        x = self.transformer1(x, lengths)
        x = self.pooling(x.transpose(1, 2))
        x = self.flatten(x)
        
        x = self.classifier(x)
        
        pred_labels = torch.argmax(x, dim=1)
        return x, pred_labels
    
    def loss(self, model_out, targets: torch.Tensor):
        predictions, _ = model_out
        targets_one_hot_encoded = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot_encoded = targets_one_hot_encoded.to(torch.float32)
        return self.loss_function(predictions, targets_one_hot_encoded)
