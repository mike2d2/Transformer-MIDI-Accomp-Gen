import pickle
import random
import sys

from dataset import create_accomp_drum_dataset
from preprocessing.etl import MidiTokenizer
import torch

from train import PAD_IDX, BOS_IDX, EOS_IDX, generate_square_subsequent_mask
saved_midi_path = 'saved_midi/predicted.mid'

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
MODEL_FILE = 'model.pt'

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
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
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    # grab random song from training set
    # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    src = src_sentence.view(-1,1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens, start_symbol=BOS_IDX).flatten()
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

def main():

    try:
        mode = sys.argv[1] # set to "train", "test", or "val" from command line
    except:
        mode = 'train'

    # load from saved models
    model = torch.load(f'{MODEL_FOLDER}/{MODEL_FILE}')

    for i in range(10):
        #try:

        print('Randomly selecting song')
        dataset_path = f'data_cleaned/spliced_318_{mode}.txt'
        vocab_size = 318
        max_seq = 2048
        train_dataset = create_accomp_drum_dataset(dataset_path, max_seq=2048, vocab_size=vocab_size)

        rand_index = random.randint(0, 1000)
        (accomp, metadata), drums = train_dataset[rand_index]
        translated_drums = translate(model, accomp)
        title = metadata['title'].replace(' ', '')

        translated_drums = remove_extra_drum_tokens(translated_drums)
        detokenize(translated_drums, metadata['ticks'], f'{mode}_transdrums_{title}')
        
        drums = remove_extra_drum_tokens(drums.tolist())

        detokenize(drums, metadata['ticks'], f'{mode}_ogdrums_{title}')
        accomp = remove_extra_accomp_tokens(accomp.tolist())

        detokenize_accomp(accomp, metadata['ticks'], f'{mode}_accomp_{title}')

        detokenize_both(accomp, drums, metadata['ticks'], f'{mode}_ogboth_{title}')

        detokenize_both(accomp, translated_drums, metadata['ticks'], f'{mode}_transboth_{title}')
         
        #except:
        #    print('error translating original drums back to midi')

if __name__ == "__main__":
    main()
