import torch
import torch.nn as nn
import torchaudio.models
import torch.nn.functional as F

class ConformerModel(nn.Module):
    def __init__(self, num_classes: int):
        super(ConformerModel, self).__init__()
        self.num_classes = num_classes
        
        self.conformer_layers = nn.Sequential(
            torchaudio.models.Conformer(
                input_dim=80,  # Example value, adjust based on preprocessing
                num_heads=8,   # Self-attention heads
                ffn_dim=2048,  # Feed-forward network dimension
                num_layers=12, # Number of Conformer layers
                num_classes=num_classes  # Number of classes
            )
        )
        
        self.pooling = nn.AdaptiveAvgPool2d((1, None))
        feature_size = 128
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 64),  # Adjust the input dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.conformer_layers(x)
        x = self.pooling(x).squeeze(dim=2)
        x = self.classifier(x)
        return x
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        targets_one_hot_encoded = F.one_hot(targets, num_classes=self.num_classes)
        return self.loss_function(predictions, targets_one_hot_encoded)
