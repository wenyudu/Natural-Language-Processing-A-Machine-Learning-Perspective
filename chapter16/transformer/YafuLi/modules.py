import torch
import torch.nn as nn
import numpy as np
import copy, math
import torch.nn.functional as F
from torch.autograd import Variable
from utils import clones, subsequent_mask
"""
Author: Yafu Li
Organization: Westlake University
Email: yafuly@gmail.com

This is a basic implementaion of transformer.

Credit: https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # embedding for source input words
        self.tgt_embed = tgt_embed # embedding for target input words, also default embedding for output prediction words
        self.generator = generator # generate distribution for prediction words
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    "Generic N layer encoder with masking."
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below).
    Implementation for equation 16.19-16.26.
    """
    def __init__(self, size, self_attn, feed_forward, dropout, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.normalize_before = normalize_before
        self.self_attn_layer_norm = LayerNorm(size)
        self.final_layer_norm = LayerNorm(size)
        self.dropout_module = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Each layer follows a Layernorm(x+Sublayer(x)) style, where Sublayer is a function such as self-attention or feedforward. Recent research shows the pre-norm version, i.e., (x+Sublayer(Layernorm(x))), converges faster than the post-norm one. Therefore, pre-norm is favoured here, which is also the same for decoder layers. Post-norm is used if 'normalize_before' is set false.
        """
        # Encoder layer forward computation
        # self-attention sublayer, equation 16.19-16.24
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # feedforward sublayer, equation 16.25-16.26
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, encoder-attn, and feed forward (defined below). 
    Implementation for equation 16.30-16.32.
    """
    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout, normalize_before=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.encoder_attn = encoder_attn
        self.feed_forward = feed_forward
        self.normalize_before = normalize_before
        self.self_attn_layer_norm = LayerNorm(size)
        self.encoder_attn_layer_norm = LayerNorm(size)
        self.final_layer_norm = LayerNorm(size)
        self.dropout_module = nn.Dropout(dropout)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."     
        # Encoder layer forward computation
        # self-attention sublayer, equation 16.30
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # encoder-attention sublayer, equation 16.31
        residual = x
        m = memory
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, m, m, src_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        # feedfoward sublyer, equation 16.32
        residul = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.feed_forward(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention computation.
    Implementaion for eqaution 16.19-16.22.
    """
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Multi-head attention computation, implementaion for eqaution 16.19-16.22
        # Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # Apply attention on all the projected vectors in batch. 
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)  
        x, self.attn = torch.matmul(p_attn, value), p_attn
        # Concat using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        # Implementaion for equation 16.28.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)