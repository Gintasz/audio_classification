import torch
import torch.nn as nn
from torchaudio.transforms import Resample, MFCC # type: ignore
import torch.nn.functional as F
from torchaudio.functional import compute_deltas  # type: ignore
from torchaudio.transforms import ComputeDeltas
import numpy as np

def pad_sequence(sequence: torch.Tensor, max_len: int, value: int = 0) -> torch.Tensor:
    # Assumes sequence is a 2D torch.Tensor (batch_size, num_samples)
    pad_size = max_len - sequence.size(1)
    if pad_size > 0:
        padded_sequence = F.pad(sequence, (0, pad_size), 'constant', value)
        return padded_sequence    
    else:
        return sequence

def normalize_waveform(waveform: torch.Tensor, target_rms: float = 1) -> torch.Tensor:
    current_rms = torch.sqrt(torch.mean(waveform**2))
    gain = target_rms / (current_rms + 1e-6)
    normalized_waveform = waveform * gain
    return normalized_waveform

def standardize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    mean = waveform.mean()
    std = waveform.std() + 1e-6  # To prevent division by zero
    standardized_waveform = (waveform - mean) / std
    return standardized_waveform


def add_noise_to_mfcc(mfcc, noise_level=0.005):
    noise = np.random.normal(0, noise_level, mfcc.shape)
    noisy_mfcc = mfcc + noise
    return noisy_mfcc

def normalise_mfcc_features(mfcc: torch.Tensor, min: int, max: int):
    normalised_mfcc = (mfcc - min) / (max - min)
    return normalised_mfcc

class AudioPreprocessingLayer(nn.Module):
    def __init__(self, input_freq: int, resample_freq: int, n_mfcc: int, max_duration_ms: int) -> None:
        super(AudioPreprocessingLayer, self).__init__()
        self.n_mfcc = n_mfcc
        self.max_length = int(max_duration_ms / 1000 * resample_freq)
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)        
        self.mfcc_transform = MFCC(
            sample_rate=resample_freq,
            n_mfcc=n_mfcc,
        )
        self.global_min = None
        self.global_max = None
        self.global_mean = None
        self.global_std = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.dim() == 2, "waveform is expected to be of shape (batch_size, num_samples)"
        # waveform = normalize_waveform(waveform, 0.5)
        # waveform = standardize_waveform(waveform)
        # resampled = self.resample(waveform)
        padded_waveform = pad_sequence(waveform, self.max_length)
        assert padded_waveform.size(1) <= self.max_length, "padded sequence length is greater than max length"

        
        mfcc = self.mfcc_transform(padded_waveform)
        # if self.global_min is not None and self.global_max is not None:
            # mfcc = normalise_mfcc_features(mfcc, self.global_min.min(), self.global_max.max())
        # mfcc = add_noise_to_mfcc(mfcc, noise_level=0.005)
        if self.global_mean is not None and self.global_std is not None:
            mfcc = (mfcc - self.global_mean.view(1, -1, 1)) / (self.global_std.view(1, -1, 1) + 1e-6)
        mfcc_delta = ComputeDeltas()(mfcc)
        mfcc_delta2 = ComputeDeltas()(mfcc_delta)        
        mfcc_with_derivatives = torch.cat((mfcc, mfcc_delta, mfcc_delta2), dim=1)
        
        
        return mfcc_with_derivatives
