#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Start writing code here...
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


# In[2]:


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs.view(1, 1, self.input_size), hidden)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1+int(self.bidirectional), 1, self.hidden_size), 
        torch.zeros(1+int(self.bidirectional), 1, self.hidden_size))


# In[3]:


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = nn.Linear(hidden_size + output_size, 1)
        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size)
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size),
        torch.zeros(1,1, self.output_size))

    def forward(self, decoder_hidden, encoder_outputs, input):
        weights = []
        for i in range(len(encoder_outputs)):
            print(decoder_hidden[0][0].shape)
            print(encoder_outputs[0].shape)
            weights.append(self.attn(torch.cat((decoder_hidden[0][0],
            encoder_outputs[i]), dim=1)))
        
        normalized_weights = F.softmax(torch.cat(weights, 1), 1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), encoder_outputs.view(1, -1, self.hidden_size))

        input_lstm = torch.cat((attn_applied[0], input[0]), dim=1)

        output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        output = self.final(output[0])

        return output, hidden, normalized_weights


# In[4]:


bidirectional = True
c = Encoder(10, 20, bidirectional)
a, b = c.forward(torch.randn(10), c.init_hidden())
print(a.shape)
print(b[0].shape)
print(b[1].shape)

x = AttentionDecoder(20 * (1+bidirectional), 25, 30)
y, z, w = x.forward(x.init_hidden(), torch.cat((a,a)), torch.zeros(1, 1, 30))
print(y.shape)
print(z[0].shape)
print(z[1].shape)
print(w)


# In[ ]:




