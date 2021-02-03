from torch import nn as nn
import torch
from torch.nn import functional as F
from arch.base import gen_order_embeds
from arch.base import l2norm
from arch.layers import RGCNLayer
from thexp import Logger

log = Logger()


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.is_embed:
            self.ft_embed = nn.Linear(sum(self.config.dim_fts), self.config.dim_embed)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, fts, *args):
        """
        Args:
          fts: size=(batch, ..., sum(dim_fts))
        Returns:
          embeds: size=(batch, dim_embed)
        """
        embeds = fts
        if self.config.is_embed:
            embeds = self.ft_embed(embeds)
        if self.config.nonlinear:
            embeds = F.relu(embeds)
        if self.config.norm:
            embeds = l2norm(embeds)
        embeds = self.dropout(embeds)
        return embeds


class FlatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)

        dim_fts = sum(self.config.dim_fts)
        self.node_embedding = nn.Embedding(self.config.num_node_types, dim_fts)

        self.register_buffer('attr_order_embeds',
                             torch.FloatTensor(gen_order_embeds(20, dim_fts)))

    def forward(self, fts, node_types, attr_order_idxs):
        '''
        Args:
          fts: size=(batch, seq_len, dim_ft)
          node_types: size=(batch, seq_len)
          attr_order_idxs: size=(batch, seq_len)
        Returns:
          embeds: size=(batch, seq_len, dim_embed)
        '''
        node_embeds = self.node_embedding(node_types)
        node_embeds = node_embeds + self.attr_order_embeds[attr_order_idxs]

        inputs = fts * node_embeds
        embeds = self.encoder(inputs)

        return embeds


class RGCNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.embed_first:
            self.first_embedding = nn.Sequential(
                nn.Linear(self.config.dim_input, self.config.dim_hidden),
                nn.ReLU())

        self.layers = nn.ModuleList()
        dim_input = self.config.dim_hidden if self.config.embed_first else self.config.dim_input
        for _ in range(self.config.num_hidden_layers):
            h2h = RGCNLayer(dim_input, self.config.dim_hidden, self.config.num_rels,
                            activation=F.relu, dropout=self.config.dropout)
            dim_input = self.config.dim_hidden
            self.layers.add_module('rgcn_{}'.format(_), h2h)

    def forward(self, attn_fts, rel_edges, *args):
        if self.config.embed_first:
            attn_fts = self.first_embedding(attn_fts)

        for i, layer in enumerate(self.layers):
            attn_fts = layer(attn_fts, rel_edges)

        return attn_fts


class RoleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rgcn_encoder = RGCNEncoder(config)

        self.node_embedding = nn.Embedding(config.num_node_types,
                                           config.dim_input)

        self.register_buffer('attr_order_embeds',
                             torch.FloatTensor(gen_order_embeds(20, self.config.dim_input)))

    def forward(self, attn_fts, node_types, attr_order_idxs, rel_edges, *args):
        """Args:
          (num_src_nodes = num_tgt_nodes)
          - attn_fts: (batch_size, num_src_nodes, in_feat)
          - rel_edges: (num_rels, num_tgt_nodes, num_src_nodes)
          - node_types: (batch_size, num_src_nodes)
          - attr_order_idxs: (batch_size, num_src_nodes)
        """
        node_embeds = self.node_embedding(node_types)
        node_embeds = node_embeds + self.attr_order_embeds[attr_order_idxs]

        input_fts = attn_fts * node_embeds

        return self.rgcn_encoder(input_fts, rel_edges)
