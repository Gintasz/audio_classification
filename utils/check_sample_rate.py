import os
import librosa

def print_sample_rates(folder_path):
    sample_rates = []
    durations = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.wav'):
                print(f'Checking sample rate and duration of {file_path}')
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    duration_ms = librosa.get_duration(y=audio, sr=sr) * 1000
                    sample_rates.append(sr)
                    durations.append(duration_ms)
                except Exception as e:
                    print(f'Error checking sample rate and duration of {file_path}: {str(e)}')

    unique_sample_rates = set(sample_rates)
    unique_durations = set(durations)
    print(f'Unique sample rates: {unique_sample_rates}')
    print(f'Unique durations: {unique_durations}')

print_sample_rates('speech_commands')

