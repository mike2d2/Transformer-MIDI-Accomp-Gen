import time
from matplotlib import pyplot as plt
import numpy as np
from dataset import create_accomp_drum_dataset
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from model.model_v1 import DrumTransformer  # Useful for optimizer and lr scheduler (Adam, SGD, Adagrad)
from model.model_v2 import Seq2SeqTransformer

if torch.cuda.is_available():
    dev = 'cuda:0'
elif torch.backends.mps.is_available():
    dev = 'mps'
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

SRC_VOCAB_SIZE = 320 #len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = 320 #len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 318, 304, 305

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

from dataset import create_accomp_drum_dataset

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    dataset_path = './data_cleaned/spliced_318_1000_train.txt'
    train_dataset = create_accomp_drum_dataset(dataset_path, max_seq=2048, vocab_size=318)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0) # removed n_workers, shuffle=True caused index to be >1000 may need different seed?


    for (src, _), tgt in train_dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # src and target are both of shape (seq_len, batch_size) so need to transpose from dataloader
        # TODO: revert these so input to model is (batch_size, seq_len)
        src = torch.transpose(src,0,1)
        tgt = torch.transpose(tgt,0,1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

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

    dataset_path = './data_cleaned/spliced_318_1000_val.txt'
    val_dataset = create_accomp_drum_dataset(dataset_path, max_seq=2048, vocab_size=318)
    
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

    for (src, _), tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

# Training Loop
from timeit import default_timer as timer
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
        torch.save(transformer, 'saved_models/model.pt')
        val_loss = 0
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

# main
# def main():
#     dataset_path = './data_cleaned/default_10000_318.txt'
#     batch_size = 2

#     num_drum_types = 6
#     vocab_size = 318 # for silence and shift tokens
#     d_ff = 512
#     input_dim = 322  # size of your vocabulary, i.e., the number of unique tokens in your MIDI data.
#     output_dim = 322 # vocab_size  # size of vocabulary as well since model will be generating token sequences of same length as input.
#     d_model = 512  # this determines the dimensionality of the model and is usually set to be in the range of 128-512,
#     #  depending on the complexity of the task. You can experiment with different values
#     nhead = 8  # number of attention heads in the model and can be set to 4-8 for most tasks.
#     num_layers = 6  # number of layers in the encoder stack, which can be set to be around 4-8.
#     dropout = 0.1  # probability of an element being zeroed in output of a layer, which can be set to 0.1-0.3 to prevent overfitting.
#     learning_rate = 1e-4  # lr that paper specifies
#     lr_step_size = 5  # number of epochs after which the learning rate is reduced, which can be set to around 5-10.
#     lr_gamma = 0.5  # factor by which lr is reduced, which can be set to 0.5-0.9 depending on the rate of convergence.
#     num_epochs = 300  # number that paper specifies
#     # device = 'cuda'  # CPU or GPU that model should be trained on. If GPU available, set this to "cuda" to enable GPU acceleration.
#     if torch.cuda.is_available():
#         dev = 'cuda:0'
#     # elif torch.backends.mps.is_available():
#     #     dev = 'mps'
#     else:
#         dev = 'cpu'
#     device = torch.device(dev)
#     print(f"Using device: {device}")

#     (train_dataset, val_dataset, test_dataset) = create_accomp_drum_dataset(dataset_path, max_seq=2048, vocab_size=vocab_size)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # removed n_workers, shuffle=True caused index to be >1000 may need different seed?
#     # val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     # test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # model = Transformer(num_layers, d_model, nhead, d_ff , dropout, vocab_size, vocab_size)
#     model = DrumTransformer(input_dim, input_dim, d_model, nhead, num_layers, dropout)
#     # model = SimpleTransformer(input_dim, output_dim, d_model)
#     model.to(device)

#     # Define optimizer and learning rate scheduler
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

#     # Define loss function
#     criterion = nn.CrossEntropyLoss()

#     # -------------------------------------------------------------------------------
#     # Train model
#     # -------------------------------------------------------------------------------
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         # Train
#         # train_epoch(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler, args.print_modulus)
        
#         # masked_data = mask_data(train_dataset.data, np.random.randint(0, input_dim))
#         # shuffled_idxs = torch.randperm(len(masked_data))
#         for i, batch in enumerate(train_loader):
#             # Get the inputs and targets
#             inputs, targets = batch[0], batch[1]
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             # outputs = model(inputs, targets[:-1])
#             outputs = model(inputs,targets)

#             # Compute the loss
#             # loss = criterion(outputs.reshape(-1, output_dim), targets[1:].reshape(-1))
#             loss = criterion(outputs.reshape(-1, output_dim), targets.view(-1))

#             # Backward pass and optimize
#             loss.backward()
#             optimizer.step()

#             # Update the running loss
#             running_loss += loss.item()

#             # Print statistics every n mini-batches
#             n = 100
#             if i % n == n - 1:
#                 print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / n:.3f}')
#                 running_loss = 0.0

#         # Update the learning rate
#         scheduler.step()

#         # Save the trained model
#         torch.save(model, 'saved_models/model.pt')

# def mask_data(src, column, instrument_percentage=0.4, timestep_percentage=.2):
#         length = len(src)
#         mask_length = int(length * instrument_percentage)
#         instrument_indices = torch.randperm(length)[:mask_length]
#         mask_length = int(length * timestep_percentage)
#         timestep_indicies = torch.randperm(length)[:mask_length]
#         src_cpy = src.copy()
#         src_cpy[instrument_indices, column] = 0
#         src_cpy[timestep_indicies] = 0

#         return src_cpy


if __name__ == "__main__":
    main()
