from matplotlib import pyplot as plt
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model.model_v3 import FiLMSeq2SeqTransformer
from dataset import create_accomp_drum_dataset


######################################################################
# Get device info

if torch.cuda.is_available():
    dev = 'cuda:0'
#elif torch.backends.mps.is_available():
#    dev = 'mps'
else:
    dev = 'cpu'
DEVICE = torch.device(dev)
print(f"Using device: {DEVICE}")


######################################################################
# During training, we need a subsequent word mask that will prevent the model from looking into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# Let's now define the parameters of our model and instantiate the same. Below, we also
# define our loss function which is the cross-entropy loss and the optimizer used for training.
#

torch.manual_seed(0)

SRC_VOCAB_SIZE = 320  # len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = 320  # len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 6
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 318, 304, 305

transformer = FiLMSeq2SeqTransformer(
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


######################################################################
# Training loop logic


def train_epoch(model, optimizer):

    model.train()
    losses = 0
    dataset_path = './data_cleaned/spliced_318_film_2_train.txt'
    train_dataset = create_accomp_drum_dataset(dataset_path, max_seq=2048, vocab_size=318)

    # removed n_workers, shuffle=True caused index to be >1000 may need different seed?
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for src, tgt in train_dataloader:

        film = src[:, -5:].to(DEVICE)
        src = src[:, :-5].to(DEVICE)
        tgt = tgt.to(DEVICE)

        # src and target are both of shape (seq_len, batch_size) so need to transpose from dataloader
        # TODO: revert these so input to model is (batch_size, seq_len)
        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, film)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):

    model.eval()
    losses = 0

    dataset_path = './data_cleaned/spliced_318_film_2_val.txt'
    val_dataset = create_accomp_drum_dataset(dataset_path, max_seq=2048, vocab_size=318)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for src, tgt in val_dataloader:

        film = src[:, -5:].to(DEVICE)
        src = src[:, :-5].to(DEVICE)
        tgt = tgt.to(DEVICE)

        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, film)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


######################################################################
# Main training loop

NUM_EPOCHS = 100


def main():
    train_loss_history = []
    val_loss_history = []
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        # Save the trained model
        torch.save(transformer, 'saved_models/model_film_splice.pt')
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        # Save the train and validation loss to a file
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        with open('losses.txt', 'a') as f:
            f.write(f"Epoch {epoch}: Train loss={train_loss:.3f}, Val loss={val_loss:.3f}\n")

        # Plot the train and validation loss history and save it to a file
        plt.plot(train_loss_history, label='Train loss')
        plt.plot(val_loss_history, label='Val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')


if __name__ == "__main__":
    main()
