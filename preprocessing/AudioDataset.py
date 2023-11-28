# labels = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero" ]
import os
import hashlib
import csv
from torch.utils.data import Dataset
import torchaudio # type: ignore
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class AudioDataset(Dataset):
    def __init__(self, dataset_path: str, enable_transform_cache: bool, included_classes: List[str], transform: Optional[nn.Module] = None):
        self.enable_transform_cache = enable_transform_cache
        self.transform = transform
        
        data: List[Tuple[str, str]] = []
        with open(dataset_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] in included_classes:
                    data.append((row[0], row[1]))
            
        unique_labels = set([row[1] for row in data])
        unique_labels_ids = {label: index for index, label in enumerate(unique_labels)}
        self.data = [(row[0], unique_labels_ids[row[1]]) for row in data]
        
        self.cache_dir = 'cache_preprocessing'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.global_min, self.global_max, self.global_mean, self.global_std = self.calculate_global_stats()


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, label = self.data[idx]
        md5_hash = hashlib.md5(audio_path.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, md5_hash)

        if self.enable_transform_cache and os.path.exists(cache_path):
            item = torch.load(cache_path)
        else:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.float()
            transformed = self.transform(waveform) if self.transform else waveform
            transformed = transformed.squeeze(0) # remove the first dimension
            transformed = transformed.transpose(0, 1)
            item = transformed, label
            if self.enable_transform_cache:
                torch.save(item, cache_path)
            
        return item
    
    def get_example_audio(self, label_id: int) -> torch.Tensor:
        for idx, (audio_path, label) in enumerate(self.data):
            if label == label_id:
                mfcc, label = self.__getitem__(idx)
                return mfcc
    
    def calculate_global_stats(self):
        all_mfccs = []
        for idx in range(len(self)):
            mfcc, _ = self.__getitem__(idx)
            all_mfccs.append(mfcc)
        
        all_mfccs = torch.cat(all_mfccs, dim=0)
    
        global_min = all_mfccs.min(dim=0)[0][:self.transform.n_mfcc]
        global_max = all_mfccs.max(dim=0)[0][:self.transform.n_mfcc]
        global_mean = all_mfccs.mean(dim=0)[:self.transform.n_mfcc]
        global_std = all_mfccs.std(dim=0)[:self.transform.n_mfcc]
        return global_min, global_max, global_mean, global_std
