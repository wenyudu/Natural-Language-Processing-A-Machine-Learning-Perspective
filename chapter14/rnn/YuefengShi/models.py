import torch.nn as nn
import torch
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout, hidden_dim, output_dim):
        
        super().__init__()
        self.d = dropout 

        ## define embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        ## define rnn layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        ## emb(x) = Wx , (see equation (13.4))
        embedded = self.embedding(text)
        ## h_t = RNN_STEP(x_t, h_(t-1)) = f(W^h * h_(t-1) + W^x * x_t + b) (see equation(14.1))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

    
    
    
