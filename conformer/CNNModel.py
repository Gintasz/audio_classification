import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_labels(labels: torch.Tensor, epsilon: float = 0.1):
    """Applies label smoothing. Default epsilon is 0.1."""
    # For binary classification, the labels are 0 and 1.
    return labels * (1 - epsilon) + (1 - labels) * epsilon

class CNNModel(nn.Module):
    def __init__(self,
            num_classes: int,
            activation_function: str,
            dropout2d_rate: float,
            skip_connections: bool,
            smooth_labels_epsilon: float):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.smooth_labels_epsilon = smooth_labels_epsilon
        self.skip_connections = skip_connections
        
        supported_activation_functions = ["relu", "silu"]
        assert activation_function in supported_activation_functions, f"activation_function {activation_function} is not supported. Supported activation functions are: {supported_activation_functions}"
        if activation_function == "relu":
            self.activation_function = nn.ReLU(inplace=True)
        elif activation_function == "silu":
            self.activation_function = nn.SiLU()
        
        
        if self.skip_connections:
            self.adjust_dim11 = nn.Conv2d(1, 128, kernel_size=1)
            self.adjust_dim12 = nn.Conv2d(128, 256, kernel_size=1)
        
        #SkipConnection, DepthwiseSeparableConv2d
        self.feature_extractor_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, padding=1),
            nn.BatchNorm2d(64),
        )
        self.feature_extractor_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=10, padding=1),
            nn.BatchNorm2d(128),
        )
        self.feature_extractor_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=10, padding=1),
            nn.BatchNorm2d(256),
        )
        self.feature_extractor_end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), # fully connected layer
            self.activation_function,
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.loss_function = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        # feature extraction
        if self.skip_connections:
            residual = x
        x = self.feature_extractor_1(x)
        x = self.activation_function(x)
        x = self.feature_extractor_2(x)
        if self.skip_connections:
            residual = self.adjust_dim11(residual)
            x = x + residual
        x = self.activation_function(x)
        x = self.feature_extractor_3(x)
        x = self.activation_function(x)
        x = self.feature_extractor_end(x)
        
        # classification
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        label = torch.argmax(probs, dim=1)
        
        return probs, label

    def loss(self, model_out, targets: torch.Tensor):
        predictions, _ = model_out
        targets_one_hot_encoded = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot_encoded = targets_one_hot_encoded.to(torch.float32)
        return self.loss_function(predictions, targets_one_hot_encoded)
    
        # logits, _ = model_out
        
        # if self.smooth_labels_epsilon > 0:
        #     labels = smooth_labels(labels, self.smooth_labels_epsilon)
        # if logits.dim() == 1:
        #     logits = logits.unsqueeze(-1)
        # return self.loss_function(logits.squeeze(1), labels.float())

