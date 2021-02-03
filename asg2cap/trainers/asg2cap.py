"""
Templete
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thexp import Trainer, Meter
from torch.nn import functional as F

from trainers import GlobalParams
from trainers.mixin import *
from thexp.utils.timing import timeit


class BaseTrainer(callbacks.BaseCBMixin,
                  datasets.BaseSupDatasetMixin,
                  models.AsgModelMixin,
                  acc.AccCider, acc.AccBleu,
                  losses.CaptionLoss,
                  Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: GlobalParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        logits = self.model(batch_data)

        meter.Lall = meter.Lall + self.loss_cap_(logits, batch_data['caption_ids'], batch_data['caption_masks'],
                                                 meter=meter, name='Lcap')

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        meter.update(timeit.meter(ratio=False))

        meter = timeit.meter()
        timeit.clear()
        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(torch.rand(xs.shape[0], params.n_classes))

    def test_eval_logic(self, dataloader, params: GlobalParams):
        params.topk = params.default([1, 5])
        from data.transforms import Int2Sent
        int2sent_func = Int2Sent()
        meter = Meter()
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                self.logger.inline(i, len(dataloader))
                if params.greedy_or_beam:
                    from thextra.inferences import sample_decode as decode_func
                else:
                    from thextra.inferences import beam_search_decode as decode_func

                enc_outs = self.model.encode(batch_data=batch_data)  # type:dict
                init_words = torch.zeros(batch_data['attn_masks'].size(0), dtype=torch.int64).to(self.device)

                states = []
                for _ in range(2):  # (hidden, cell)
                    states.append(torch.zeros((2, params.batch_size, self.params.hidden_size),
                                              dtype=torch.float32).to(self.device))

                _, max_attn_len = batch_data['attn_masks'].shape
                prev_attn_score = torch.zeros((params.batch_size, max_attn_len), device=self.device)
                prev_attn_score[:, 0] = 1

                enc_globals = enc_outs['init_states']
                enc_memories = enc_outs['attn_fts']
                enc_masks = batch_data['attn_masks']
                flow_edges = batch_data['flow_edges']

                result = decode_func(init_words, self.model.decoder.step_fn, params.max_words_in_sent,
                                     greedy=True, states=states, enc_globals=enc_globals,
                                     enc_memories=enc_memories, memory_masks=enc_masks,
                                     prev_attn_score=prev_attn_score, flow_edges=flow_edges)

                if params.greedy_or_beam:
                    pred_sent, word_logprobs = result
                    sent_pool = []
                    for sent, word_logprob in zip(pred_sent, word_logprobs):
                        sent_pool.append([(word_logprob.sum().item(), sent, word_logprob)])
                else:
                    sent_pool = result
                    pred_sent = [pool[0][1] for pool in sent]

                word_sents = []
                gt_sents = []
                for i, sent, gt_sent in zip(batch_data['index'], pred_sent, batch_data['caption_ids']):
                    sent = sent.detach().cpu().numpy().tolist()
                    gt_sent = gt_sent.detach().cpu().numpy().tolist()

                    sent = int2sent_func(sent)
                    gt_sent = int2sent_func(gt_sent)
                    gt_sents.append(gt_sent)
                    word_sents.append(sent)

                gt_sents = {i: [v] for i, v in enumerate(gt_sents)}
                word_sents = {i: [v] for i, v in enumerate(word_sents)}

                self.acc_bleu_(gt_sents, word_sents, meter=meter)
                self.acc_cider_(gt_sents, word_sents, meter=meter)

        return meter


if __name__ == '__main__':
    params = GlobalParams()
    # params.device = 'cuda:0'
    params.from_args()

    trainer = BaseTrainer(params)
    trainer.train()
    trainer.save_model()
