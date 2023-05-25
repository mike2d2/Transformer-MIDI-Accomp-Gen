# Rythmic Complexity Index
import librosa
import numpy as np

def get_onset_times(audio_file):
    y, sr = librosa.load(audio_file)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times

audio_file = ''
onset_times = get_onset_times(audio_file)

def compute_iois(onset_times):
    return np.diff(onset_times)

iois = compute_iois(onset_times)

def create_ioi_histogram(iois, num_bins=50):
    hist, _ = np.histogram(iois, bins=num_bins, density=True)
    return hist

ioi_histogram = create_ioi_histogram(iois)

def compute_rhythmic_complexity_index(ioi_histogram):
    # Normalize the histogram to ensure it represents a probability distribution
    prob_dist = ioi_histogram / ioi_histogram.sum()

    # Compute Shannon entropy
    rci = -np.sum(prob_dist * np.log2(prob_dist + np.finfo(float).eps))
    return rci

rci = compute_rhythmic_complexity_index(ioi_histogram)
print("Rhythmic Complexity Index:", rci)
