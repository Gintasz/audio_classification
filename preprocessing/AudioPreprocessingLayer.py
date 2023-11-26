import torch
import torch.nn as nn
from torchaudio.transforms import Resample, Loudness, Spectrogram, ComputeDeltas, AmplitudeToDB, InverseSpectrogram, MFCC # type: ignore
import torch.nn.functional as F
from torchaudio.functional import compute_deltas  # type: ignore
import numpy as np
from noisereduce.torchgate import TorchGate as TG # type: ignore
import pyloudnorm as pyln # type: ignore
from audiomentations import Compose, AddBackgroundNoise, PolarityInversion, BandPassFilter, TanhDistortion, AddGaussianNoise, AddShortNoises, ApplyImpulseResponse, Gain, PitchShift, TimeStretch


import warnings
warnings.filterwarnings('ignore', message='Possible clipped samples in output.', module='pyloudnorm')

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

def pre_emphasize(waveform, alpha=0.97):
    return torch.cat((waveform[:, :1], waveform[:, 1:] - alpha * waveform[:, :-1]), dim=1)

def level_volume(waveform, sample_rate: int):
    waveform_np = waveform.numpy().T
    waveform_np = pyln.normalize.peak(waveform_np, -1.0)
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(waveform_np)
    loudness_normalized_waveform_np = pyln.normalize.loudness(waveform_np, loudness, -12.0)
    loudness_normalized_waveform = torch.Tensor(loudness_normalized_waveform_np).T
    return loudness_normalized_waveform

def add_noise_to_mfcc(mfcc, noise_level=0.005):
    noise = np.random.normal(0, noise_level, mfcc.shape)
    noisy_mfcc = mfcc + noise
    return noisy_mfcc

def normalise_mfcc_features(mfcc: torch.Tensor, min: int, max: int):
    normalised_mfcc = (mfcc - min) / (max - min)
    return normalised_mfcc

class AudioPreprocessingLayer(nn.Module):
    def __init__(self, input_freq: int, resample_freq: int, n_mfcc: int, max_duration_ms: int, augment: bool) -> None:
        super(AudioPreprocessingLayer, self).__init__()
        self.n_mfcc = n_mfcc
        self.max_length = int(max_duration_ms / 1000 * resample_freq)
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)        
        self.mfcc_transform = MFCC(
            sample_rate=resample_freq,
            n_mfcc=n_mfcc,
        )
        self.noise_reduction_torchgate = TG(sr=resample_freq, nonstationary=False)
        
        self.augment = Compose([
            ApplyImpulseResponse(
                ir_path="speech_commands/_impulse_responses_",
                p=0.3
            ),
            # AddBackgroundNoise(
            #     sounds_path="speech_commands/_background_noise_",
            #     min_snr_in_db=3.0,
            #     max_snr_in_db=30.0,
            #     # noise_transform=PolarityInversion(p=0.5),
            #     p=1
            # ),
            BandPassFilter(p=0.3),
            AddGaussianNoise(min_amplitude=0.00005, max_amplitude=0.0001, p=0.3),
            TanhDistortion(p=0.3),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.3),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3, leave_length_unchanged=True),
        ])
        
        self.global_min = None
        self.global_max = None
        self.global_mean = None
        self.global_std = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.dim() == 2, "waveform is expected to be of shape (batch_size, num_samples)"
        if self.augment:
            waveform_np = waveform.numpy()
            waveform_np = self.augment(samples=waveform_np, sample_rate=self.resample.new_freq)
            waveform = torch.from_numpy(waveform_np)
            # DEBUG:
            #import sounddevice as sd
            #sd.play(waveform_np.T, samplerate=self.resample.new_freq)
        waveform = self.resample(waveform)
        waveform = self.noise_reduction_torchgate(waveform)
        assert not torch.any(torch.isnan(waveform)), "No element after noise_reduction_torchgate should be nan"
        waveform = level_volume(waveform, self.resample.new_freq)
        waveform = pre_emphasize(waveform)
        # waveform = normalize_waveform(waveform, 0.5)
        padded_waveform = pad_sequence(waveform, self.max_length)
        assert padded_waveform.size(1) <= self.max_length, "padded sequence length is greater than max length"

        
        mfcc = self.mfcc_transform(padded_waveform)
        if self.global_mean is not None and self.global_std is not None:
            mfcc = (mfcc - self.global_mean.view(1, -1, 1)) / (self.global_std.view(1, -1, 1) + 1e-6)
            
        mfcc_delta = ComputeDeltas()(mfcc)
        mfcc_delta2 = ComputeDeltas()(mfcc_delta)        
        mfcc_with_derivatives = torch.cat((mfcc, mfcc_delta, mfcc_delta2), dim=1)
        
        return mfcc_with_derivatives
