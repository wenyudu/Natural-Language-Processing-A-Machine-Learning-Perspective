import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
from modules import *
from utils import NoamOpt, get_std_opt, LabelSmoothing
"""
Author: Yafu Li
Organization: Westlake University
Email: yafuly@gmail.com

This is a toy example to demonstrate transformer-based neural machine translation.

Data: a subset extracted from iwslt16 de-en dataset, which consists of 2,481 sentences for trainning, 200 sentences for validation and a vocab of 665 tokens.
"""

MAX_LEN = 10 # max sentence length for padding

# initialize model
def make_model(src_vocab, tgt_vocab, N=3, 
               hidden_size=64, d_ff=256, h=4, dropout=0.3):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, hidden_size)
    ff = PositionwiseFeedForward(hidden_size, d_ff, dropout)
    position = PositionalEncoding(hidden_size, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(hidden_size, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(hidden_size, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(hidden_size, src_vocab), c(position)),
        nn.Sequential(Embeddings(hidden_size, tgt_vocab), c(position)),
        Generator(hidden_size, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


# build vocab
with open('vocab.subset', 'r') as v:
    lines = v.readlines() 
    vocab = {}
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    for i,l in enumerate(lines):
        vocab[l.split("\t")[0]] = i+4

# loss for training and validation
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

# data holder
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = \
                self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# run through data for one epoch
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, 
                            batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 5 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                        (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    
    return total_loss / total_tokens

# simple greedy decoding method
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        if next_word == vocab['<eos>']: # stop looping when an EOS token is generated 
            break
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# read sample data
def data_gen(V, batch, is_valid=False):
    prefix = 'valid' if is_valid else 'train'
    with open('%s.de'%prefix, 'r') as de, open('%s.en'%prefix, 'r') as en:
        lines_de = de.readlines()
        lines_en = en.readlines()

    # map txt to id and augment padding
    ids_de = []
    for l in lines_de:
        ids = [vocab['<sos>']]+[vocab[w] for w in l.strip().split()]+[vocab['<eos>']]
        ids = ids + [vocab['<pad>']]*(MAX_LEN-len(ids))
        ids_de.append(ids)
    ids_en = []
    for l in lines_en:
        ids = [vocab['<sos>']]+[vocab[w] for w in l.strip().split()]+[vocab['<eos>']]
        ids = ids + [vocab['<pad>']]*(MAX_LEN-len(ids))
        ids_en.append(ids)

    assert len(lines_de) == len(lines_en)
    size = len(lines_de)
    nbatches = size // batch

    for i in range(nbatches-1):
        data_src = torch.tensor(ids_de[i*batch:(i+1)*batch])
        data_tgt = torch.tensor(ids_en[i*batch:(i+1)*batch])
        src = Variable(data_src, requires_grad=False)
        tgt = Variable(data_tgt, requires_grad=False)
        yield Batch(src, tgt, 0)
    # remaining data
    data_src = torch.tensor(ids_de[i*batch:])
    data_tgt = torch.tensor(ids_en[i*batch:])
    yield Batch(src, tgt, 0)



# trainning and validation
V = len(vocab)
bsz = 64
epochs = 20
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(epochs):
    print("EPOCH %d: " % epoch)
    print("training ...")
    model.train()
    loss = run_epoch(data_gen(V, bsz), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    print("Validating ...")
    model.eval()
    print("Valid loss: %f" % float(run_epoch(data_gen(V, bsz, is_valid=True), model, 
                    SimpleLossCompute(model.generator, criterion, None))))


# inference 
test_sent = 'vielen dank .' # sentence to translate, with tokens seperated by space
ids = [vocab['<sos>']]+[vocab[w] for w in test_sent.strip().split()]+[vocab['<eos>']]
ids = ids + [vocab['<pad>']]*(MAX_LEN-len(ids))
model.eval()
src = Variable(torch.LongTensor([ids]) )
src_mask = (src != vocab['<pad>']).unsqueeze(-2)
hyp_ids=greedy_decode(model, src, src_mask, max_len=32, start_symbol=1)
hyp = []
r_vocab = dict((y,x) for x,y in vocab.items())
for i in list(hyp_ids[0][1:]):
    hyp.append(r_vocab[int(i)])
print(" ".join(hyp))

