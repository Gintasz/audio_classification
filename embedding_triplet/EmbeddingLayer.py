import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size: int):
        super(EmbeddingLayer, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers for producing embeddings
        self.fc1 = nn.Linear(64 * 4 * 13, 128)
        self.fc2 = nn.Linear(128, embedding_size)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, batch_distances: torch.Tensor, labels: torch.Tensor):
        # Assuming x is of shape (batch_size, 1, 32, 13) - batch_size, channels, freq_bins, time_steps
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 4 * 13)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 normalize the embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x