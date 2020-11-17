import torch.nn as nn
import torch
import torch.nn.functional as F

    
    
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()
        
        # Load embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define converlutional layer
        self.convs = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                                kernel_size = (filter_sizes, embedding_dim)) 
        # Define Fully-Connected layer
        self.fc = nn.Linear(n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        text = text.permute(1, 0)
        ## emb(x) = Wx , (see equation 13.4)
        embedded = self.embedding(text) ## load embedding (batch_size * input length * embedding size)
        embedded = embedded.unsqueeze(1) ## Add input channals (batch_size * input channal size *input length * embedding size)
        ## converlutional layer (batch_size * * input channal size * input length * embedding size)
        ## H_1^(n-K+1) = CNN(X_(1:n), K, d_O) = W⊗X_(1:n)+b, ⊗ means converlutional operation, see equation (13.16)
        conved = F.relu(self.convs(embedded)).squeeze(3)
        ## after converlutional layer (batch_size * out_channal_size *[(input_length - filter_size) + 1] )
        ## maximum pooling (batch_size * out_channal_size *[(input_length - filter_size) + 1] )
        ## max(X_(1:N)) = <max(x_i)[1], max(x_i)[2], ...,max(x_i)[d]>,  max pooling, see equation(13.15)
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        ## after maximum pooling (batch_size * out_channal_size)
        cat = self.dropout(pooled)
        ## full connect
        ## o = W^o * h + b^o, see equation (13.18)
        return self.fc(cat)
    
 