# -------------------------------------------------------------------------------
# Define Transformer Model Architecture
# -------------------------------------------------------------------------------

"""
The Transformer class defines the architecture of the transformer model:
- input_dim  specifies the dimensionality of the input token sequence,
- output_dim specifies the dimensionality of the output token sequence,
- d_model    specifies the dimensionality of the transformer layers,
- nhead      specifies the number of attention heads in the transformer layers,
- num_layers specifies the number of transformer layers,
- dropout    specifies the dropout probability.
The PositionalEncoding class defines the positional encoding used by the transformer layers:
- d_model    specifies the dimensionality of the transformer layers,
- dropout    specifies the dropout probability,
- max_len    specifies the maximum length of the input sequence.
init_weights method initializes the weights of the embedding and output layers.
forward      method processes the input token sequence and generates the output token sequence.
The input token sequence is first embedded using an embedding layer,
    then processed using a positional encoding layer,
    and then passed through the transformer layers.
The output of the transformer layers is then passed through a linear layer to generate the output token sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # useful to apply activation function like F.relu() (ReLU, softmax, sigmoid) to
#  output or calculate a loss function like F.cross_entropy() during training.
import math
import numpy as np

import torch
import torch.nn as nn


class DrumTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers=2, dropout=0.1):
        self.is_training = True
        super(DrumTransformer, self).__init__()
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(output_dim, d_model)
        self.src_pos_encoder = PositionalEncoding(d_model, dropout)
        self.tgt_pos_encoder = PositionalEncoding(d_model, dropout)

        # self.transformer = nn.Transformer(
        #         d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
        #         num_decoder_layers=1, dropout=self.dropout, # activation=self.ff_activ,
        #         dim_feedforward=self.d_ff, custom_decoder=self.dummy
        # )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, output_dim)

    def init_weights(self):
        init_range = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=init_range)

    def forward(self, src, target):
        src = self.src_embedding(src)
        src = self.src_pos_encoder(src)
        target = self.tgt_embedding(target)
        target = self.tgt_pos_encoder(target)

        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(target, memory)
        logits = self.decoder(output)
        # probs = F.softmax(logits, dim=1)
        # probs = F.softmax(logits, dim=2)
        probs = logits

        return probs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0), :]
        return self.dropout(x)


# -------------------------------------------------------------------------------
# Loads the data from a file,
#   converts each token sequence to a PyTorch LongTensor,
#   and stores them as a list.
# -------------------------------------------------------------------------------
import torch.utils.data as data


class PercussionDataset(data.Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                tokens = [int(token) for token in line.strip().split()]
                self.data.append(tokens)

    # Returns a single sequence at a time.
    def __getitem__(self, index):
        return torch.LongTensor(self.data[index])

    # Returns the total number of sequences in the dataset.
    def __len__(self):
        return len(self.data)


