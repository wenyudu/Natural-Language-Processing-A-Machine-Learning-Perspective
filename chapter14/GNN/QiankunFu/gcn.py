import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, adjacency, input_feature):
        # Eq. 14.31
        support = torch.mm(input_feature, self.W)
        output = torch.mm(adjacency.float(), support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        # Eq. 14.32
        x = F.relu(self.gcn1(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn2(adj, x))
        return x


if __name__ == '__main__':
    # set parameters for grn
    device = torch.device('cpu')
    gnn_dropout = 0.1
    node_num = 4
    embed_dim = 8
    gnn_hidden_dim = 8
    gnn_layer_num = 4
    # create your graph data [x, adjacent matrix], there is some random example data
    x = torch.rand(node_num, embed_dim).to(device)
    adj = torch.randint(0, 2, [node_num, node_num]).to(device)
    # build a 2-layer GCN model
    gcn = GCN(embed_dim, embed_dim, gnn_hidden_dim, gnn_dropout).to(device)

    # get the final node representation for the downstream task like classification etc.
    node_reps = gcn(x, adj)

    print("output node representation:\n", node_reps)
    print("Done! We can use the reps for downstream task.")

