import pickle as pkl
import random
import sys

from dataset import create_accomp_drum_dataset
from preprocessing.etl import MidiTokenizer
import torch

from train import PAD_IDX, BOS_IDX, EOS_IDX, generate_square_subsequent_mask
saved_midi_path = 'saved_midi/predicted_film.mid'

if torch.cuda.is_available():
    dev = 'cuda:0'
elif torch.backends.mps.is_available():
    dev = 'mps'
else:
    dev = 'cpu'
DEVICE = torch.device(dev)
print(f"Using device: {DEVICE}")

MIDI_FOLDER = 'saved_midi/'
MODEL_FOLDER = 'saved_models/'
MODEL_FILE = 'model_film.pt'

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, film_toks):

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    film_toks = film_toks.to(DEVICE)

    memory = model.encode(src, src_mask, film_toks)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask, film_toks)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

        if next_word == EOS_IDX:
            break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, film_toks):
    model.eval()
    # grab random song from training set
    # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    src = src_sentence.view(-1,1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens, start_symbol=BOS_IDX, film_toks=film_toks).flatten()
    return tgt_tokens[1:num_tokens].tolist()

def detokenize(midi_tokens, ticks, filename):
    tokenizer = MidiTokenizer()
    tokenizer.compile_drums(midi_tokens, ticks_per_beat=ticks, out_path=f'{MIDI_FOLDER}{filename}.mid')

def detokenize_accomp(accomp_tokens, ticks, filename):
    tokenizer = MidiTokenizer()
    tokenizer.compile_accompaniment(accomp_tokens, ticks_per_beat=ticks, out_path=f'{MIDI_FOLDER}{filename}.mid')

def detokenize_both(accomp_tokens, drum_tokens, ticks, filename):
    tokenizer = MidiTokenizer()
    tokenizer.compile_both(drum_tokens, accomp_tokens, ticks_per_beat=ticks, out_path=f'{MIDI_FOLDER}{filename}.mid')

def remove_extra_accomp_tokens(tokenized_accomp):
    #print(tokenized_midi)
    #print(len(tokenized_midi))
    res = [i for i in tokenized_accomp if i not in [PAD_IDX, BOS_IDX, EOS_IDX, 319]]
    #res = [i for i in res if i != 282]
    #print(len(res))
    return res

def remove_extra_drum_tokens(tokenized_drums):
    #print(tokenized_midi)
    #print(len(tokenized_midi))
    res = [i for i in tokenized_drums if i < 304]
    #print(len(res))
    return res

def write_info(filename, metadata, index):
    with open(f'{MIDI_FOLDER}{filename}', "w") as f:
        # write lines to the file
        f.write(f'Dataset index: {index}\n')
        f.write('genres:\n')
        for item in metadata:
            f.write("%s\n" % item)
		
def main():

    with open('preprocessing/film_feats.pkl', 'rb') as f:
        [GENRES, _, TICKS] = pkl.load(f)
    print(GENRES)
    mode = sys.argv[1]  # set to "train", "test", or "val" from command line
    num_genres = len(sys.argv) - 2      # add up to 3 genres
    genres = []
    for i in range(num_genres):
        if i >= 3:
            break
        genres.append(sys.argv[i + 2])
    if 1 <= len(genres) < 3:
        genres += ['none'] * (3 - len(genres))
    genre_toks = torch.LongTensor([GENRES.index(f"b'{gen}'") for gen in genres])

    # load from saved models
    model = torch.load(f'{MODEL_FOLDER}/{MODEL_FILE}')
    custom_idx = [339, 453, 287, 22, 56, 345 ,384,89, 203, 45]
    for i in range(10):
        print('Selecting song')
        filename_base = f'{mode}_'    
        dataset_path = f'data_cleaned/spliced_318_film_{mode}.txt'
        vocab_size = 318
        max_seq = 2048
        train_dataset = create_accomp_drum_dataset(dataset_path, max_seq=max_seq, vocab_size=vocab_size)
        rand_index = None
        if custom_idx:
            rand_index = custom_idx[i]
        else:
            rand_index = random.randint(0, len(train_dataset))
        input_seq, drums = train_dataset[rand_index]
        
        accomp = input_seq[:-5]
        metadata_tokens = input_seq[-5:]
        if genres:
            metadata_tokens[0:3] = genre_toks
            filename_base = f'{filename_base}{genres[0]}_'
        translated_drums = translate(model, accomp, metadata_tokens)

        ticks_tok = metadata_tokens[-1]
        ticks_val = int(TICKS[ticks_tok])

        translated_drums = remove_extra_drum_tokens(translated_drums)
        detokenize(translated_drums, ticks_val, f'{filename_base}transdrums_{i}')
        
        drums = remove_extra_drum_tokens(drums.tolist())

        detokenize(drums, ticks_val, f'{filename_base}ogdrums_{i}')
        accomp = remove_extra_accomp_tokens(accomp.tolist())

        detokenize_accomp(accomp, ticks_val, f'{filename_base}accomp_{i}')

        detokenize_both(accomp, drums, ticks_val, f'{filename_base}ogboth_{i}')

        detokenize_both(accomp, translated_drums, ticks_val, f'{filename_base}transboth_{i}')
        print(genres)        
        write_info(f'{filename_base}info_{i}.txt', genres, rand_index) 
        #except:
        #    print('error translating original drums back to midi')

if __name__ == "__main__":
    main()
