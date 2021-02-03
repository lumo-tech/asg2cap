import numpy as np
import torch
from arch import encoder, layers
from torch import nn
from thexp import Logger

log = Logger()


class ASGModel(nn.Module):

    def __init__(self, role_encoder, multi_rel_encoder, decoder, **kwargs):
        super().__init__()

        self.role_encoder = role_encoder
        self.multi_rel_encoder = multi_rel_encoder
        self.decoder = decoder

    # def build_submods(self):
    #     submods = {}
    #     submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
    #     submods[ATTNENCODER] = controlimcap.encoders.gcn.RoleEncoder(self.config.subcfgs[ATTNENCODER])
    #     submods[DECODER] = controlimcap.decoders.cfattention.ContentFlowAttentionDecoder(
    #         self.config.subcfgs[DECODER])
    #     return submods

    # def prepare_input_batch(self, batch_data, is_train=False):
    #     outs = super().prepare_input_batch(batch_data, is_train=is_train)
    #     outs['node_types'] = torch.LongTensor(batch_data['node_types']).to(self.device)
    #     outs['attr_order_idxs'] = torch.LongTensor(batch_data['attr_order_idxs']).to(self.device)
    #     return outs

    def encode(self, batch_data):
        attn_embeds = self.role_encoder(batch_data['attn_fts'],
                                        batch_data['node_types'], batch_data['attr_order_idxs'],
                                        batch_data['rel_edges'])

        graph_embeds = torch.sum(attn_embeds * batch_data['attn_masks'].unsqueeze(2), 1)
        graph_embeds = graph_embeds / torch.sum(batch_data['attn_masks'], 1, keepdim=True)
        enc_states = self.multi_rel_encoder(torch.cat([batch_data['mp_fts'], graph_embeds], 1))
        return {'init_states': enc_states, 'attn_fts': attn_embeds}

    def decode(self, batch_data, enc_outs):
        logits = self.decoder(batch_data['caption_ids'][:, :-1],
                              enc_outs['init_states'],
                              enc_outs['attn_fts'],
                              batch_data['attn_masks'], batch_data['flow_edges'])

        return logits

    def forward(self, batch_data):
        enc_outs = self.encode(batch_data)

        logits = self.decode(batch_data, enc_outs)

        return logits
