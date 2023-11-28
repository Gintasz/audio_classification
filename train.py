import torch
import torch.nn as nn
from preprocessing.AudioDataset import AudioDataset
from torch.utils.data import DataLoader
from preprocessing.AudioPreprocessingLayer import AudioPreprocessingLayer
from prototype_learning.OnePerClassPrototypeModel import OnePerClassPrototypeModel
from prototype_learning.OnePerClassPrototypeModel2 import OnePerClassPrototypeModel2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
if __name__ == '__main__':
    print(f"Cuda available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = ["up", "down", "left", "right"]
    
    # Load TRAINING dataset
    dataset_train = AudioDataset(dataset_path="dataset_train.txt", enable_transform_cache=True, included_classes=labels, transform=AudioPreprocessingLayer(
        input_freq = 16000, resample_freq = 16000, n_mfcc = 13, max_duration_ms = 1000, augment=False
    ))
    dataset_train.transform.global_min = dataset_train.global_min
    dataset_train.transform.global_max = dataset_train.global_max
    dataset_train.transform.global_mean = dataset_train.global_mean
    dataset_train.transform.global_std = dataset_train.global_std
    dataset_train.transform.augment = True
    dataloader_train = DataLoader(dataset_train, batch_size=1000, shuffle=True, num_workers=os.cpu_count())
    
    # Load VALIDATION dataset
    dataset_validate = AudioDataset(dataset_path="dataset_validate.txt", enable_transform_cache=True, included_classes=labels, transform=AudioPreprocessingLayer(
        input_freq = 16000, resample_freq = 16000, n_mfcc = 13, max_duration_ms = 1000, augment=False
    ))
    dataset_validate.transform.global_mean = dataset_train.transform.global_mean
    dataset_validate.transform.global_std = dataset_train.transform.global_std
    dataloader_validate = DataLoader(dataset_validate, batch_size=1000, shuffle=True, num_workers=os.cpu_count())

    # Initialize model
    model = OnePerClassPrototypeModel().to(device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    num_epochs = 1000
    validate_epoch_frequency = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    best_loss = float('inf')
    for index, epoch in enumerate(range(num_epochs)):
        # TRAIN
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_correct = 0
        batch_i = 0
        
        for batch_x, labels in dataloader_train:
            batch_x = batch_x.to(device)
            labels = labels.to(device)
            print(f"Batch {batch_i+1} / {len(dataloader_train)}")
            optimizer.zero_grad()
            
            model_out = model.forward(batch_x)
            loss = model.loss(model_out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_i += 1

            # Calculate accuracy
            _, pred_y = model_out
            total_correct += torch.sum(pred_y == labels).item()

        avg_loss = total_loss / len(dataloader_train)
        accuracy = total_correct / len(dataset_train)
        print(f'Epoch {epoch+1}/{num_epochs}, average Loss: {avg_loss}, Accuracy: {accuracy}')
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'model.pt')
            print(f'Model saved to file.')
            
        # VALIDATE
        if epoch > 0 and epoch % validate_epoch_frequency == 0:
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
            print(f'Epoch {epoch+1}/{num_epochs} VALIDATION, average Loss: {avg_val_loss}, Accuracy: {accuracy}')
            