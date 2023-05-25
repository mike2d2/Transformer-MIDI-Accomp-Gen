import zipfile
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
# Pattern Consistency

def euclidean_distance(pattern1, pattern2):
    return np.sum(np.array(pattern1) != np.array(pattern2))

def binarize_onset_locations(onset_locations, beat_resolution=16):
    binary_onset_locations = []
    for bar in onset_locations:
        binary_pattern = np.zeros(beat_resolution)
        for loc in bar:
            index = int(loc * beat_resolution) % beat_resolution
            binary_pattern[index] = 1
        binary_onset_locations.append(binary_pattern)
    return binary_onset_locations


def pattern_consistency(midi_data1, midi_data2, beat_resolution=16):
    # Extract onset locations and binarize the patterns for all bars in both MIDI files
    onset_locations1 = extract_onset_locations_percussion(midi_data1)
    binary_onset_locations1 = binarize_onset_locations(onset_locations1, beat_resolution)

    onset_locations2 = extract_onset_locations_percussion(midi_data2)
    binary_onset_locations2 = binarize_onset_locations(onset_locations2, beat_resolution)

    # Calculate the distances between corresponding bar pairs of the two MIDI files
    min_bars = min(len(binary_onset_locations1), len(binary_onset_locations2))
    distances = []
    for i in range(min_bars):
        distance = euclidean_distance(binary_onset_locations1[i], binary_onset_locations2[i])
        distances.append(distance)

    return distances


midi_files = []  # choose files



def extract_onset_locations(midi_data, beat_resolution=16):
    onset_locations = []

    for instrument in midi_data.instruments:
        # Check if the instrument is a percussion instrument
        if instrument.is_drum:
            for note in instrument.notes:
                # Calculate the onset location as a fraction of a beat
                onset_time = note.start
                beat_position = midi_data.time_to_tick(onset_time) % beat_resolution
                onset_location = beat_position / beat_resolution
                onset_locations.append(onset_location)

    return onset_locations

def extract_onset_locations_percussion(midi_data, beat_resolution=16):
    percussion_onsets = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                onset = note.start
                percussion_onsets.append(onset)

    percussion_onsets.sort()
    bars = np.array(midi_data.get_downbeats()[1:]) - np.array(midi_data.get_downbeats()[:-1])
    barwise_onsets = []

    for i, bar_length in enumerate(bars):
        start_time = midi_data.get_downbeats()[i]
        end_time = midi_data.get_downbeats()[i+1]
        onsets_in_bar = [onset for onset in percussion_onsets if start_time <= onset < end_time]
        barwise_onsets.append(onsets_in_bar)

    return barwise_onsets

d = []
lengths = []
for i in range(10):
    distances = []
    with zipfile.ZipFile('/content/drive/MyDrive/566 Project Datasets/saved_midi/spliced_train_318_1000/spliced_train_1000_318.zip', 'r') as zip_ref:
        file_names = zip_ref.namelist()
        with zip_ref.open(file_names[i]) as file1:
            midi_data1 = pretty_midi.PrettyMIDI(file1)
            with zip_ref.open(file_names[i+10]) as file2:
                midi_data2 = pretty_midi.PrettyMIDI(file2)
                with zip_ref.open(file_names[i+20]) as file3:
                    midi_data3 = pretty_midi.PrettyMIDI(file3)
                    distances1 = pattern_consistency(midi_data1, midi_data2)
                    distances2 = pattern_consistency(midi_data2, midi_data3)
                    if len(distances1) == 9:
                        d.append(np.array(distances1))
                    if len(distances2) == 9:
                        d.append(np.array(distances2))
                    lengths.append(len(distances))

distances = np.array(d)
distances = distances.sum(0).tolist()
print(distances)



def count_and_normalize_onset_locations(onset_locations, beat_resolution=16):
    # Count the occurrences of each onset location
    counts = np.zeros(beat_resolution)
    for loc in onset_locations:
        index = int(loc * beat_resolution)
        counts[index] += 1

    # Normalize the counts
    normalized_counts = counts / np.sum(counts)

    return normalized_counts

normalized_counts_list = []

for midi_file in midi_files:
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    onset_locations = extract_onset_locations(midi_data)
    normalized_counts = count_and_normalize_onset_locations(onset_locations)
    normalized_counts_list.append(normalized_counts)


x = np.arange(0, 1, 1/16)
colors = ['blue', 'green', 'red']  # Change the colors based on the number of MIDI files

for i, normalized_counts in enumerate(normalized_counts_list):
    plt.plot(x, normalized_counts, label=f'MIDI File {i+1}', color=colors[i])

plt.xlabel('Onset Location')
plt.ylabel('Frequency')
plt.title('Improvised Bars: Distribution of Onset Locations')
plt.xticks(np.arange(0, 1, 0.25), ['0', '1/4', '1/2', '3/4'])
plt.legend()
plt.grid()
plt.show()
