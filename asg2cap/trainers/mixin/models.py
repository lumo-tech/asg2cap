from torch import nn

from thexp import Trainer
from thexp.contrib import EMA, ParamGrouper

import arch
from .. import GlobalParams
import torch


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

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            if self.params.ema:
                model = self.ema_model
            else:
                model = self.model
            if not self.params.with_fc:
                return model.fc(model(xs))
            return model(xs)

    def models(self, params: GlobalParams):
        encoder, attn_encoder = load_encoder(params)
        decoder = load_decoder(params)

        from arch.gcns import ASGModel

        model = ASGModel(attn_encoder, encoder, decoder, params=params)

        if params.distributed:
            from torch.nn.modules import SyncBatchNorm
            model = SyncBatchNorm.convert_sync_batchnorm(model.cuda())
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank])
        else:
            self.model = model

        if params.ema:
            self.ema_model = EMA(self.model)

        grouper = ParamGrouper(self.model)
        # noptim = params.optim.args.copy()
        param_groups = [
            grouper.create_param_group(attn_encoder.parameters(), **params.optim.args),
            grouper.create_param_group(encoder.parameters(), **params.optim.args),
            grouper.create_param_group(decoder.parameters(), **params.optim.args),
        ]

        self.optim = params.optim.build(param_groups)
        if not params.distributed:
            self.to(self.device)
