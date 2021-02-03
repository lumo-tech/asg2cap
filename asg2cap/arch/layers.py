import torch.nn as nn
import torch

from thexp import Logger

log = Logger()


# import framework.configbase
# import framework.ops


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,
                 bias=None, activation=None, dropout=0.0):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation

        self.loop_weight = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, attn_fts, rel_edges):
        '''Args:
          attn_fts: (batch_size, max_src_nodes, in_feat)
          rel_edges: (batch_size, num_rels, max_tgt_nodes, max_srt_nodes)
        Retunrs:
          node_repr: (batch_size, max_tgt_nodes, out_feat)
        '''
        loop_message = torch.einsum('bsi,ij->bsj', attn_fts, self.loop_weight)
        loop_message = self.dropout(loop_message)

        neighbor_message = torch.einsum('brts,bsi,rij->btj', rel_edges, attn_fts, self.weight)

        node_repr = loop_message + neighbor_message
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)

        return node_repr
