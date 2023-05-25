import linecache
import json
import pickle as pkl

import torch
from torch.utils.data import Dataset


class DrumsAccompanimentDataset(Dataset):

    def __init__(self, file_path, max_seq, vocab_size, num_classes=512, ) -> None:

        self.data = []
        self.file_path = file_path
        self.num_classes = num_classes
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        self.max_drum_len = max([
            len(json.loads(line)['drums']) for line in open(self.file_path, 'r')
        ])
        with open('preprocessing/film_feats.pkl', 'rb') as f:
            genres, time_sigs, ticks = pkl.load(f)
            self.film_vocab = {
                'genres': genres,
                'time_sigs': time_sigs,
                'ticks': ticks
            }

    def __len__(self):

        lines = 0
        with open(self.file_path, 'r') as f:
            for _ in f:
                lines += 1

        return lines
    
    def __getitem__(self, index):
        # index = 0 causes error linecache.getline(self.file_path, 0) returns empty string
        index = index if index != 0 else 1
        # indices = torch.randperm(self.data_size)[:self.batch_size]
        
        # Get event arrays
        item_dict = json.loads(linecache.getline(self.file_path, index))
        drum_events = torch.LongTensor(item_dict['drums'])
        accomp_events = torch.LongTensor(item_dict['accomp'])

        # Get metadata for genres
        genres = item_dict['genres']
        meta_genres = []
        for g in genres:
            if g in self.film_vocab['genres']:
                meta_genres.append(self.film_vocab['genres'].index(g))
            else:
                meta_genres.append(self.film_vocab['genres'].index('none'))

        # Get metadata for time sig
        time_sig = item_dict['timesig']
        if time_sig in self.film_vocab['time_sigs']:
            meta_timesig = [self.film_vocab['time_sigs'].index(time_sig)]
        else:
            meta_timesig = [self.film_vocab['time_sigs'].index('none')]

        # Get metadata for ticks
        ticks = str(item_dict['ticks'])
        if ticks in self.film_vocab['ticks']:
            meta_ticks = [self.film_vocab['ticks'].index(ticks)]
        else:
            meta_ticks = [self.film_vocab['ticks'].index('none')]

        # Compile metadata
        metadata_seq = torch.LongTensor(meta_genres + meta_timesig + meta_ticks)

        # Add pad tokens to sequences
        if len(accomp_events) < 2048:
            accomp_events = torch.cat((accomp_events, torch.LongTensor([319]*(2048 - len(accomp_events)))))
        if len(drum_events) < self.max_drum_len:
            drum_events = torch.cat((drum_events, torch.LongTensor([319]*(self.max_drum_len - len(drum_events)))))
        elif len(drum_events) > self.max_drum_len:
            drum_events = drum_events[0:self.max_drum_len]

        # Put metadata together with accompaniment temporarily
        input_seq = torch.cat((accomp_events, metadata_seq))

        return input_seq, drum_events
    
    def select_seq(self, accomp_events, drum_events, seq_len, vocab_size):
        TOKEN_START = vocab_size+1
        TOKEN_PAD = vocab_size+2
        TOKEN_END = vocab_size+3
        SEQUENCE_START = 0
        accomp_tensor = torch.full((seq_len,), TOKEN_PAD)
        drum_tensor = torch.full((seq_len,), TOKEN_PAD)
        accomp_tensor[0] = TOKEN_START
        drum_tensor[0] = TOKEN_START

        if len(accomp_events) < seq_len:
            accomp_tensor[len(accomp_events)-1] = TOKEN_END
            accomp_tensor[1:len(accomp_events)-1] = accomp_events[1:len(accomp_events)-1]
        else:
            accomp_tensor[seq_len-1] = TOKEN_END
            accomp_tensor[1:seq_len-1] = accomp_events[1:seq_len-1]

        if len(drum_events) < seq_len:
            drum_tensor[len(drum_events)] = TOKEN_END
            drum_tensor[:len(drum_events)] = drum_events
        else:
            drum_tensor[seq_len-1] = TOKEN_END
            drum_tensor[1:seq_len-1] = drum_events[1:seq_len-1]

        return accomp_tensor, drum_tensor
    
def select_seq(accomp_events, drum_events, seq_len, vocab_size):
    TOKEN_PAD_ACCOMP = vocab_size
    TOKEN_END_ACCOMP = vocab_size+1
    TOKEN_PAD_DRUM = TOKEN_PAD_ACCOMP - 12
    TOKEN_END_DRUM = TOKEN_END_ACCOMP - 12
    SEQUENCE_START = 0
    accomp_tensor = torch.full((seq_len,), TOKEN_PAD_ACCOMP)
    drum_tensor = torch.full((seq_len,), TOKEN_PAD_DRUM)

    if len(accomp_events) < seq_len:
        accomp_tensor[len(accomp_events)] = TOKEN_END_ACCOMP
        accomp_tensor[:len(accomp_events)] = accomp_events
    else:
        accomp_tensor[seq_len-1] = TOKEN_END_ACCOMP
        accomp_tensor[1:seq_len-1] = accomp_events[1:seq_len-1]

    if len(drum_events) < seq_len:
        drum_tensor[len(drum_events)] = TOKEN_END_DRUM
        drum_tensor[:len(drum_events)] = drum_events
    else:
        drum_tensor[seq_len-1] = TOKEN_END_DRUM
        drum_tensor[1:seq_len-1] = drum_events[1:seq_len-1]
    
    return accomp_tensor, drum_tensor
    
    
def create_accomp_drum_dataset(dataset_path, max_seq=2048, random_seq=False, vocab_size=None):

    dataset = DrumsAccompanimentDataset(dataset_path, max_seq=max_seq, vocab_size=vocab_size)
    print('length of dataset: ',len(dataset))

    # train_subset, val_subset, test_subset = torch.utils.data.random_split(dataset, [800, 100, 100])
    train_dataset = dataset

    # TODO: Get train, val, test splits
    return train_dataset
