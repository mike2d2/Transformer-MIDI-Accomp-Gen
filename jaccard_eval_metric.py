# NOTE: Need to put folder "saved_midi" containing the original ("original_drums_"+str(i)+".mid") and predicted 
#       ("saved_midi/translated_drums_"+str(j)+".mid") midi files in the same folder as this file.

from midiutil.MidiFile import MIDIFile
import os
import pretty_midi
import pdb; 

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

#------------------------------------------------------------------------
# FUNCTIONS:
#------------------------------------------------------------------------

# Save confusion matrix heat map.
def save_heatmap_img(confusion_matrix, title="Confusion Matrix Heat Map", file_name="heatmap.png"):
    # row_sums = np.sum(confusion_matrix, axis=1)
    # normalized_array = confusion_matrix / row_sums

    df = pd.DataFrame(confusion_matrix, index = [i for i in "0123456789"], columns = [i for i in "0123456789"])
    plt.figure(figsize=(10,7))
    sn.heatmap(df, annot=True,)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(file_name)


def read_midi(file_path):
    """
    Reads MIDI file and returns a set of note events
    """
    midi_data = pretty_midi.PrettyMIDI(file_path)

    note_events = set()
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note_events.add((note.pitch, note.start))

    return note_events

# Calculates the Jaccard similarity between two sets
def jaccard_similarity(set1, set2):

    pitch_set1 = set(note[0] for note in set1)
    pitch_set2 = set(note[0] for note in set2)
    intersection_size = len(pitch_set1.intersection(pitch_set2))
    union_size = len(pitch_set1.union(pitch_set2))
    return intersection_size / union_size

#------------------------------------------------------------------------
# START MAIN:
#------------------------------------------------------------------------
score_mat = []

for i in range(10):
        
    # Path to the MIDI true files
    midi_file_true = "saved_midi/original_drums_"+str(i)+".mid"

    score_list = []  

    for j in range(10):
        # Path to the MIDI predict files
        midi_file_pred = "saved_midi/translated_drums_"+str(j)+".mid"

        # Read MIDI files and get note events
        note_events_true = read_midi(midi_file_true)
        note_events_pred = read_midi(midi_file_pred)

        # Calculate Jaccard similarity
        jaccard_sim = jaccard_similarity(note_events_true, note_events_pred)
        
        score_list.append(jaccard_sim)
        # print(f"Jaccard similarity between {os.path.basename(midi_file1)} and {os.path.basename(midi_file2)}: {jaccard_sim}")

    # print("midi_file_true_"+str(i))
    # print(score_list)

    score_mat.append(score_list)
    # print("\n")
    
# print(score_mat)
save_heatmap_img(score_mat, "MIDI Jaccard Similarity Heatmap", "midi_jaccard_heatmap.png")


