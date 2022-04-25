import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


# GCN Model
class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()

        in_feats = model_config['in_feats']
        n_hidden = model_config['n_hidden']
        n_layers = model_config['n_layers']
        activation = model_config['activation']
        dropout = model_config['dropout']

        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(246, 1)

    def forward(self, graph):
        features = graph.ndata['feat']
        features = features.to(torch.float32)
        for i, layer in enumerate(self.layers):
            if i != 0:
                features = self.dropout(features)
            features = layer(graph, features)
        features = self.fc(features.mean(1))
        return features

