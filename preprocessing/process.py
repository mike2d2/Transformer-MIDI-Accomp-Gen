from etl import MidiDatasetTokenizer, MidiTokenizer
data_dir = '../data_spliced/'
dataset_name = 'spliced_318_film_3'
num_accompaniment = 4
cap = 0

tokenizer = MidiTokenizer()
dataset_tokenizer = MidiDatasetTokenizer(data_dir, tokenizer, num_accompaniment)
dataset_tokenizer.write_tokenized(dataset_name, 2048, cap)
