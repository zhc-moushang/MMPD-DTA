import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool,GATConv,global_mean_pool,GCNConv,SAGEConv
import numpy as np


d_model=60          #the number of expected features in the input
dim_feedforward = 256          #the dimension of the feedforward network model
n_heads = 2         #the number of heads in the multiheadattention models
n_layers=2          # the number of TransformerEncoderLayers in each block
class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src):
        output = src
        for mod in self.layers:
            output,attn = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output,attn
class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2,attn = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn[:,0,:]
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)
class GIN(torch.nn.Module):
    '''
    4-layer GCN model class.
    '''

    def __init__(self, c_feature=108,MLP_dim=96):
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)
        nn5 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(MLP_dim)

        self.lin = Linear(MLP_dim, 120)
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # x = self.conv4(x, edge_index)
        # x = x.relu()
        # x = self.conv5(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=0.1)
        x = self.lin(x)

        return x
class GraphSAGE(torch.nn.Module):
    def __init__(self,c_feature=108,MLP_dim=98):
        super().__init__()
        self.conv1 = SAGEConv(c_feature,MLP_dim,aggr='mean')
        self.conv2 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv3 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv4 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv5 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv6 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv7 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.lin = Linear(MLP_dim,120)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = global_add_pool(x, batch)
        x = self.lin(x)

        return x
class MMPD_DTA(torch.nn.Module):
    def __init__(self, MLP_dim=96, dropout=0.1,c_feature=108):
        super(MMPD_DTA, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.src_emb = nn.Embedding(26, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(26, d_model), freeze=True)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # GIN model for extracting ligand features
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)

        self.fc1_c = Linear(MLP_dim, 120)
        self.poc_fc = Linear(120, 60)
        self.fc1 = nn.Linear(120+120+120, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

        # FNN
        self.classifier = nn.Sequential(
            nn.Linear(120 +120 +60 , 512),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256, 1),
        )

        self.complex_graph = GraphSAGE()
    def forward(self, data):
        # print(data)
        x_ligand, edge_index_ligand , batch_ligand = data.x_t, data.edge_index_t,data.x_t_batch
        x_complex,edge_index_complex,batch_complex = data.x_s, data.edge_index_s,data.x_s_batch
        protein = data.protein

        x =  F.relu(self.conv1(x_ligand, edge_index_ligand))
        x = self.bn1(x)
        x =  F.relu(self.conv2(x, edge_index_ligand))
        x = self.bn2(x)
        x =  F.relu(self.conv3(x, edge_index_ligand))
        x = self.bn3(x)
        x =  F.relu(self.conv4(x, edge_index_ligand))
        x = self.bn4(x)
        x = global_add_pool(x, batch_ligand)
        x =  F.relu(self.fc1_c(x))
        x = self.dropout(x)

        com = self.complex_graph(x_complex,edge_index_complex,batch_complex)


        protein = self.src_emb(protein) + self.pos_emb(protein)

        embedded_xt, _ = self.transformer_encoder(protein)
        protein = embedded_xt[:, 0, :]

        x = torch.cat([protein , com,x], dim=1)
        x = self.classifier(x)

        return x
