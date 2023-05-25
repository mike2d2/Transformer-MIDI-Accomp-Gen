import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, in_features, conditioning_features):
        super(FiLM, self).__init__()
        self.scale_transform = nn.Linear(conditioning_features, in_features)
        self.shift_transform = nn.Linear(conditioning_features, in_features)

    def forward(self, x, conditioning):
        scale = self.scale_transform(conditioning)
        shift = self.shift_transform(conditioning)
        x = x * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        return x

class Generator(nn.Module):
    def __init__(self, num_tokens, embedding_size, hidden_size, num_layers, conditioning_size):
        super(Generator, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, embedding_size)
        self.positional_embedding = PositionalEncoding(embedding_size)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.conditioning_layers = nn.ModuleList([
            FiLM(hidden_size, conditioning_size)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, num_tokens)

    def forward(self, tokens, conditioning):
        x = self.token_embedding(tokens)
        x = self.positional_embedding(x)
        for transformer, conditioning_layer in zip(self.transformer_layers, self.conditioning_layers):
            x = transformer(x)
            x = conditioning_layer(x, conditioning)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)
