from etl import MidiDatasetTokenizer, MidiTokenizer

cap = 20000

tokenizer = MidiTokenizer()
dataset = MidiDatasetTokenizer('../data_raw/', tokenizer, 4)

dataset.summarize_film_feats(cap)
