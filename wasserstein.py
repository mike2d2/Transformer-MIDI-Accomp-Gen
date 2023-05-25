import zipfile
from scipy.stats import wasserstein_distance
from collections import Counter
import pretty_midi


def aggregate_feature_counts(feature_lists):
    feature_counts = Counter()
    for features in feature_lists:
        feature_counts.update(features)
    return feature_counts


def normalize_counts(counts):
    total = sum(counts.values())
    probabilities = {feature: count / total for feature, count in counts.items()}
    return probabilities


def extract_midi_features(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    features = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            onset_time = midi_data.time_to_tick(note.start)
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            feature = (onset_time, instrument_name)
            features.append(feature)

    return features


midi_files_set1 = []
midi_files_set2 = []
for i in range(10):
    with zipfile.ZipFile(
            '/content/drive/MyDrive/566 Project Datasets/saved_midi/spliced_train_318_1000/spliced_train_1000_318.zip',
            'r') as zip_ref:
        file_names = zip_ref.namelist()
        file1 = zip_ref.open(file_names[i + 0])
        file2 = zip_ref.open(file_names[i + 10])
        midi_files_set2.append(file1)
        midi_files_set2.append(file2)

all_features_set1 = [extract_midi_features(midi_file) for midi_file in midi_files_set1]
all_features_set2 = [extract_midi_features(midi_file) for midi_file in midi_files_set2]

feature_counts_set1 = aggregate_feature_counts(all_features_set1)
feature_counts_set2 = aggregate_feature_counts(all_features_set2)

feature_probabilities_set1 = normalize_counts(feature_counts_set1)
feature_probabilities_set2 = normalize_counts(feature_counts_set2)

# Combine the two dictionaries into a single one, with missing keys set to 0
combined_keys = set(feature_probabilities_set1.keys()).union(feature_probabilities_set2.keys())
probabilities1 = [feature_probabilities_set1.get(key, 0) for key in combined_keys]
probabilities2 = [feature_probabilities_set2.get(key, 0) for key in combined_keys]

# Compute the Earth Mover's Distance
emd = wasserstein_distance(probabilities1, probabilities2)
print("Earth Mover's Distance:", emd)
