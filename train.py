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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import csv
import matplotlib.pyplot as plt
import seaborn as sns


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
        
    # Load TEST dataset
    dataset_test = AudioDataset(dataset_path="dataset_test.txt", enable_transform_cache=True, cache_in_memory=True, included_classes=labels, transform=preprocessing)
    dataloader_test = DataLoader(dataset_test, batch_size=100, shuffle=True, num_workers=0)
    if not dataset_test.validate_dataset():
        print("Test dataset failed validation.")
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

    num_epochs = 1000
    validate_epoch_frequency = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

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
            # print(f"Batch {batch_i+1} / {len(dataloader_train)}")
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
                # print(f"Validation Batch {batch_i+1} / {len(dataloader_validate)}")
            
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
            
            if accuracy < best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'model.pt')
                print(f'Model saved to file.')
            
            print("Testing...")
            model.eval()
            total_test_loss = 0
            total_correct = 0
            batch_i = 0
        
            true_labels = []
            pred_labels = []
            for batch_x, labels in dataloader_test:
                batch_x = batch_x.to(device)
                labels = labels.to(device)
                # print(f"Test Batch {batch_i+1} / {len(dataloader_test)}")
            
                model_out = model.forward(batch_x)
                loss = model.loss(model_out, labels)
            
                total_test_loss += loss.item()
                batch_i += 1

                # Calculate accuracy
                _, pred_y = model_out
                total_correct += torch.sum(pred_y == labels).item()
                true_labels.append(labels.cpu())
                pred_labels.append(pred_y.cpu())
                
            true_labels = torch.cat(true_labels).numpy()
            pred_labels = torch.cat(pred_labels).numpy()

            accuracy = total_correct / len(dataset_test)
            precision = precision_score(true_labels, pred_labels, average='weighted')
            recall = recall_score(true_labels, pred_labels, average='weighted')
            conf_matrix = confusion_matrix(true_labels, pred_labels)
            tn = conf_matrix[0, 0]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]
            tp = conf_matrix[1, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            avg_test_loss = total_test_loss / len(dataloader_test)
            print(f'Epoch {epoch+1}/{num_epochs} TEST, average Loss: {avg_val_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Specificity: {specificity}')

            # Append to history_test.csv
            with open('history_test.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, accuracy, precision, recall, specificity])

            # Save confusion matrix image
            if not os.path.exists('output'):
                os.makedirs('output')
                
            plt.figure(figsize=(150,150))
            sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
            plt.title(f"Test Confusion Matrix")
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.show()
            plt.savefig(f'output/{epoch}_test_confusion.png')
            print("Confusion Matrix:")
            print(conf_matrix)
            