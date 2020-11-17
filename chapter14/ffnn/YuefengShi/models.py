import torch.nn as nn
import torch
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, hidden_dim_2, hidden_dim_3, output_dim, dropout):
        
        super().__init__()
        
        ## Loading embeddings (Number of words * )
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        ## Linear function 1 (embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(in_features = embedding_dim, out_features = hidden_dim)
        ## Non-linearity function 1
        self.relu1 = nn.ReLU()
        ## Linear function 2 (hidden_dim, hidden_dim_2)
        self.fc2 = nn.Linear(in_features = hidden_dim, out_features = hidden_dim_2)
        ## Non-linearity function 2
        self.relu2 = nn.ReLU()
        ## Linear function 3 (hidden_dim_2, hidden_dim_3)
        self.fc3 = nn.Linear(in_features = hidden_dim_2, out_features = hidden_dim_3)
        ## Non-linearity function 3
        self.relu3 = nn.ReLU()

        ## Linear function 4 (Out)
        self.fc4 = nn.Linear(in_features = hidden_dim_3, out_features = output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        embedded = self.embedding(text)
        
        ## h^1 = ReLU(W^1 * x + b^1), x means the imput vector embedded, see equation(13.9)
        output = self.fc1(embedded)
        output = self.dropout(self.relu1(output))

        ## h^2 = ReLU(W^2 * h^1 + b^2), see equation(13.9)
        output = self.fc2(output)
        output = self.dropout(self.relu2(output))

        ## h^3 = ReLU(W^3 * h^2 + b^3), see equation(13.9)
        output = self.fc3(output)
        output = self.dropout(self.relu3(output))

        ## output = W^4 * h^3
        output = self.fc4(output[-1,:,:])
        return output
    
    
    

    
 