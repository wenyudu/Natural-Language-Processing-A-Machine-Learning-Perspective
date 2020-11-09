import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Eq. 14.36
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, node_dim, hidden_dim, dropout, head_num, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GATLayer(node_dim, hidden_dim // head_num, dropout=dropout, alpha=alpha, concat=False) for _ in range(head_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        # Eq. 14.35
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # we can add our downstream task modules here
        return x


if __name__ == '__main__':
    # set parameters for grn
    device = torch.device('cpu')
    gnn_dropout = 0.1
    node_num = 4
    embed_dim = 8
    gnn_hidden_dim = 8
    head_num = 4
    gnn_layer_num = 2
    # create your graph data [x, adjacent matrix], there is some random example data
    x = torch.rand(node_num, embed_dim).to(device)
    adj = torch.randint(0, 2, [node_num, node_num]).to(device)
    # build a 2-layer GAT model
    gat = GAT(embed_dim, gnn_hidden_dim, gnn_dropout, head_num).to(device)

    # get the final node representation for the downstream task like classification etc.
    node_reps = gat(x, adj)

    print("output node representation:\n", node_reps)
    print("Done! We can use the reps for downstream task.")
