import torch
from thexp import Trainer

from arch.gcns import ASGModel
from .. import GlobalParams


def load_encoder(params: GlobalParams):
    from arch import encoder
    if params.attn_encoder == 'base':
        attn_encoder = encoder.Encoder(params)
    elif params.attn_encoder == 'rgcn':
        attn_encoder = encoder.RGCNEncoder(params)
    elif params.attn_encoder == 'role_rgcn':
        attn_encoder = encoder.RoleEncoder(params)
    else:
        assert False

    base_encoder = encoder.Encoder(params)

    return base_encoder, attn_encoder


def load_decoder(params: GlobalParams):
    from arch import decoder
    if params.decoder == 'base':
        model = decoder.Decoder(params)
    elif params.decoder == 'memory':
        model = decoder.MemoryDecoder(params)
    elif params.decoder == 'memory_flow':
        model = decoder.MemoryFlowDecoder(params)
    elif params.decoder == 'attn':
        model = decoder.AttnDecoder(params)
    elif params.decoder == 'butd_attn':
        model = decoder.BUTDAttnDecoder(params)
    elif params.decoder == 'cont_flow_attn':
        model = decoder.ContentFlowAttentionDecoder(params)
    else:
        assert False

    return model


class ModelMixin(Trainer):

    def predict(self, xs):
        raise NotImplementedError()

    def models(self, params: GlobalParams):
        raise NotImplementedError()


class AsgModelMixin(ModelMixin):
    """base end-to-end model"""

    def predict(self, batch_data) -> torch.Tensor:
        with torch.no_grad():
            model = self.model
            return model(batch_data)

    def models(self, params: GlobalParams):
        encoder, attn_encoder = load_encoder(params)
        decoder = load_decoder(params)

        self.model = ASGModel(attn_encoder, encoder, decoder, params=params)

        self.optim = params.optim.build(self.model.parameters())
        if not params.distributed:
            self.to(self.device)
