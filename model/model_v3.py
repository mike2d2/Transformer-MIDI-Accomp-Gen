######################################################################
# Seq2Seq Network using Transformer
# ---------------------------------
#
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks.
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
# Finally, the output of the Transformer model is passed through linear layer
# that gives unnormalized probabilities for each token in the target language.
#

import copy
import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    '''helper Module that adds positional encoding to the token embedding to introduce a notion of word order.'''

    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    '''helper Module to convert tensor of input indices into corresponding tensor of token embeddings'''

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class FiLM(nn.Module):
    '''Learn/predict scale and shift parameters for tensor of size <out_features>'''

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.gamma = nn.Linear(in_feats, out_feats)
        self.beta = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        gamma = self.gamma(x)
        beta = self.beta(x)
        return gamma, beta


class FiLMTransformerEncoderLayer(TransformerEncoderLayer):
    '''Single transformer encoder layer with FiLM applied'''

    def __init__(self, d_model, nhead, dim_feedforward, dropout, film_in_size):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        self.film_in_size = film_in_size
        self.film_self_attention = FiLM(film_in_size, d_model)
        self.film_ff = FiLM(film_in_size, dim_feedforward)

    def forward(self, src, src_mask, src_key_padding_mask, film_input):

        gamma_attn, beta_attn = self.film_self_attention(film_input)
        gamma_ff, beta_ff = self.film_ff(film_input)

        # Get our attention output and apply FiLM and dropout, and add residual connection, and then normalize
        src_enc, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src_enc = ((1 + gamma_attn) * src_enc) + beta_attn  # film
        src_enc = src + (self.dropout(src_enc))             # dropout and residual connection
        norm_src_enc = self.norm1(src_enc)

        # Now go through feedforward layer, and apply FiLM
        src_ff = self.linear1(norm_src_enc)
        src_ff = self.activation(src_ff)
        src_ff = self.linear2(self.dropout(src_ff))
        src_ff = ((1 + gamma_ff) * src_ff) + beta_ff    # film
        src_ff = norm_src_enc + self.dropout(src_ff)    # dropout and residual connection
        norm_src_ff = self.norm2(src_ff)

        return norm_src_ff


class FiLMTransformerDecoderLayer(TransformerDecoderLayer):
    '''Single transformer decoder layer with FiLM applied'''

    def __init__(self, d_model, nhead, dim_feedforward, dropout, film_in_size):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        self.film_in_size = film_in_size
        self.film_self_attention = FiLM(film_in_size, d_model)
        self.film_ff = FiLM(film_in_size, dim_feedforward)
        self.film_cross_attention = FiLM(film_in_size, d_model)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, film_input):

        gamma_self_attn, beta_self_attn = self.film_self_attention(film_input)
        gamma_ff, beta_ff = self.film_ff(film_input)
        gamma_cross_attn, beta_cross_attn = self.film_cross_attention(film_input)

        # Self attention on tgt
        tgt_enc, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt_enc = ((1 + gamma_self_attn) * tgt_enc) + beta_self_attn    # film
        tgt_enc = tgt + (self.dropout(tgt_enc))                         # dropout and residual conncetion
        norm_tgt_enc = self.norm1(tgt_enc)

        # Cross attention
        tgt_cross_enc, _ = self.multihead_attn(norm_tgt_enc, memory, memory, attn_mask=None, key_padding_mask=memory_key_padding_mask)
        tgt_cross_enc = ((1 + gamma_cross_attn) * tgt_cross_enc) + beta_cross_attn    # film
        tgt_cross_enc = norm_tgt_enc + self.dropout(tgt_cross_enc)              # dropout and residual connection
        norm_tgt_cross_enc = self.norm2(tgt_cross_enc)

        # Feedforward
        tgt_ff = self.linear1(norm_tgt_cross_enc)
        tgt_ff = self.activation(tgt_ff)
        tgt_ff = self.linear2(self.dropout(tgt_ff))
        tgt_ff = ((1 + gamma_ff) * tgt_ff) + beta_ff        # film
        tgt_ff = norm_tgt_cross_enc + self.dropout(tgt_ff)  # dropout and residual connectoin
        norm_tgt_ff = self.norm3(tgt_ff)

        return norm_tgt_ff


class FiLMTransformerEncoder(TransformerEncoder):
    '''Multilayer transformer encoder with FiLM applied'''

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)

    def forward(self, src, src_mask, src_key_padding_mask, film_input):

        out = src

        for mod in self.layers:
            out = mod(out, src_mask, src_key_padding_mask, film_input)

        if self.norm is not None:
            out = self.norm(out)

        return out


class FiLMTransformerDecoder(TransformerDecoder):
    '''Multilayer transformer decoder with FiLM applied'''

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, film_input):

        out = tgt

        for mod in self.layers:
            out = mod(out, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, film_input)

        if self.norm is not None:
            out = self.norm(out)

        return out


class FiLMSeq2SeqTransformer(nn.Module):
    '''Overall encoder-decoder transformer architecture with FiLM applied'''

    def __init__(
            self,
            num_encoders: int,
            num_decoders: int,
            embedding_size: int,
            nhead: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            genre_emb_size: int = 50,
            timesig_emb_size: int = 16,
            ticks_emb_size: int = 16):

        super().__init__()

        # Encoder and decoders
        film_in_size = 3*genre_emb_size + timesig_emb_size + ticks_emb_size
        encoder_layer = FiLMTransformerEncoderLayer(embedding_size, nhead, dim_feedforward, dropout, film_in_size)
        decoder_layer = FiLMTransformerDecoderLayer(embedding_size, nhead, dim_feedforward, dropout, film_in_size)
        self.encoder = FiLMTransformerEncoder(encoder_layer, num_encoders)
        self.decoder = FiLMTransformerDecoder(decoder_layer, num_decoders)

        # Token embeddings and positional embeddings for sequence
        self.generator = nn.Linear(embedding_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, embedding_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)

        # Film feature embeddings
        self.genre_embeddings = TokenEmbedding(581, genre_emb_size)
        self.timesig_embeddings = TokenEmbedding(30, timesig_emb_size)
        self.ticks_embeddings = TokenEmbedding(21, ticks_emb_size)

    def gen_film_input(self, film_feature_toks):

        if len(film_feature_toks.shape) > 1:
            batch_size = film_feature_toks.shape[0]
            genre_emb_flat = self.genre_embeddings(film_feature_toks[:, 0:3]).reshape((batch_size, -1))
            timesig_emb_flat = self.timesig_embeddings(film_feature_toks[:, 3])
            ticks_emb_flat = self.ticks_embeddings(film_feature_toks[:, 4])
            return torch.cat([genre_emb_flat, timesig_emb_flat, ticks_emb_flat], dim=1)
        else:
            genre_emb_flat = self.genre_embeddings(film_feature_toks[0:3]).flatten()
            timesig_emb_flat = self.timesig_embeddings(film_feature_toks[3])
            ticks_emb_flat = self.ticks_embeddings(film_feature_toks[4])
            return torch.cat([genre_emb_flat, timesig_emb_flat, ticks_emb_flat])

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_padding_mask, film_toks):

        # Sequence encodings
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # FiLM feature embeddings
        film_input = self.gen_film_input(film_toks)

        # Perforom the forward pass
        encoded = self.encoder(src_emb, src_mask, src_padding_mask, film_input)
        decoded = self.decoder(tgt_emb, encoded, tgt_mask, tgt_padding_mask, memory_padding_mask, film_input)
        generated = self.generator(decoded)

        return generated

    def encode(self, src, src_mask, film_toks):
        film_input = self.gen_film_input(film_toks)
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.encoder(src_emb, src_mask, None, film_input)

    def decode(self, tgt, memory, tgt_mask, film_toks):
        film_input = self.gen_film_input(film_toks)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.decoder(tgt_emb, memory, tgt_mask, None, None, film_input)
