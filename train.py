import torch
import torch.nn as nn
from preprocessing.AudioDataset import AudioDataset
from torch.utils.data import DataLoader
from preprocessing.AudioPreprocessingLayer import AudioPreprocessingLayer
from prototype_learning.OnePerClassPrototypeModel import OnePerClassPrototypeModel
from conformer.ConformerModel import ConformerModel
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
    dataloader_train = DataLoader(dataset_train, batch_size=1000, shuffle=True, num_workers=0)
    if not dataset_train.validate_dataset():
        print("Training dataset failed validation.")
        exit(1)
    
    # Load VALIDATION dataset
    dataset_validate = AudioDataset(dataset_path="dataset_validate.txt", enable_transform_cache=True, cache_in_memory=True, included_classes=labels, transform=preprocessing)
    dataloader_validate = DataLoader(dataset_validate, batch_size=1000, shuffle=True, num_workers=0)
    if not dataset_validate.validate_dataset():
        print("Validation dataset failed validation.")
        exit(1)
    
    # Load min, max, mean, std of audio features so we can standardise
    # with open('dataset_train_stats.pkl', 'wb') as f:
    #     dataset_train_stats = dataset_train.calculate_global_stats()
    #     pickle.dump(dataset_train_stats, f)
    
    # with open('dataset_train_stats.pkl', 'rb') as f:
    #     global_min, global_max, global_mean, global_std = pickle.load(f)
    #     preprocessing.global_min = global_min
    #     preprocessing.global_max = global_max
    #     preprocessing.global_mean = global_mean
    #     preprocessing.global_std = global_std

    # Initialize model
    model = ConformerModel(num_classes=len(labels)).to(device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Load existing weights if they exist
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt"))
        print("Loaded saved model weights.")

    num_epochs = 1000
    validate_epoch_frequency = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_accuracy = float('inf')
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
            if loss != 0:
                loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
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
            
        if accuracy < best_accuracy:
            best_accuracy = accuracy
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
            