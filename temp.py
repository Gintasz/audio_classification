import torch
import torch.nn as nn
from preprocessing.AudioDataset import AudioDataset
from torch.utils.data import DataLoader
from preprocessing.AudioPreprocessingLayer import AudioPreprocessingLayer
from prototype_learning.OnePerClassPrototypeModel import OnePerClassPrototypeModel
from conformer.ConformerModel import ConformerModel
from conformer.TransformerModel import TransformerModel
from conformer.CNNModel import CNNModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

if __name__ == '__main__':
    print(f"Cuda available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = ["up", "down", "left", "right", "go", "no", "stop", "yes"]
    preprocessing = AudioPreprocessingLayer(input_freq = 16000, resample_freq = 16000, n_mfcc = 13, max_duration_ms = 1000, augment=False)
    
    # Load TRAINING dataset
    dataset_train = AudioDataset(dataset_path="dataset_train.txt", enable_transform_cache=True, cache_in_memory=True, included_classes=labels, transform=preprocessing)
    dataloader_train = DataLoader(dataset_train, batch_size=100, shuffle=True, num_workers=0)
    if not dataset_train.validate_dataset():
        print("Training dataset failed validation.")
        exit(1)
    
    # Load VALIDATION dataset
    dataset_validate = AudioDataset(dataset_path="dataset_validate.txt", enable_transform_cache=True, cache_in_memory=True, included_classes=labels, transform=preprocessing)
    dataloader_validate = DataLoader(dataset_validate, batch_size=100, shuffle=True, num_workers=0)
    if not dataset_validate.validate_dataset():
        print("Validation dataset failed validation.")
        exit(1)
    
    # Initialize model
    model = CNNModel(
        num_classes=len(labels),
        activation_function='relu',
        dropout2d_rate=0,
        skip_connections=False,
        smooth_labels_epsilon=0.1
    ).to(device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Load existing weights if they exist
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt"))
        print("Loaded saved model weights.")

    print("Validating...")
    model.eval()
    total_val_loss = 0
    total_correct = 0
    batch_i = 0

    for batch_x, labels in dataloader_validate:
        batch_x = batch_x.to(device)
        labels = labels.to(device)
        print(f"Validation Batch {batch_i+1} / {len(dataloader_validate)}")
    
        model_out = model.forward(batch_x)
        loss = model.loss(model_out, labels)
    
        total_val_loss += loss.item()
        batch_i += 1

        # Calculate accuracy
        _, pred_y = model_out
        total_correct += torch.sum(pred_y == labels).item()

    accuracy = total_correct / len(dataset_validate)
    avg_val_loss = total_val_loss / len(dataloader_validate)
    print(f'VALIDATION, average Loss: {avg_val_loss}, Accuracy: {accuracy}')
    