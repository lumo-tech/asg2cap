import torch
from torch import nn
import numpy as np
import math

BOS = 0
EOS = 1
UNK = 2


def l2norm(inputs, dim=-1):
    # inputs: (batch, dim_ft)
    norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
    inputs = inputs / norm.clamp(min=1e-10)
    return inputs


def gen_order_embeds(max_len, dim_ft):
    order_embeds = np.zeros((max_len, dim_ft))
    position = np.expand_dims(np.arange(0, max_len - 1).astype(np.float32), 1)
    div_term = np.exp(np.arange(0, dim_ft, 2) * -(math.log(10000.0) / dim_ft))
    order_embeds[1:, 0::2] = np.sin(position * div_term)
    order_embeds[1:, 1::2] = np.cos(position * div_term)
    return order_embeds


def beam_search_decode(words, step_fn, max_words_in_sent,
                       beam_width=5, sent_pool_size=5, expand_fn=None, select_fn=None, **kwargs):
    '''
    Inputs are the same as sample_decode
    '''
    k = beam_width
    batch_size = words.size(0)
    # store the best sentences
    sent_pool = [[] for i in range(batch_size)]
    # remained beams for each input
    batch_sent_pool_remain_cnt = np.zeros((batch_size,)) + sent_pool_size
    # store selected words in every step to recover path
    step_words = []
    step_word_logprobs = []
    # store previous indexs of selected words
    step_prevs = []
    # sum of log probs of current sents for each beams
    cum_logprob = None  # Tensor

    # row_idxs = [[0, ..., 0], [k, ..., k], ..., [(batch-1)*k, ..., (batch-1)*k]
    row_idxs = torch.arange(0, batch_size * k, k).unsqueeze(1).repeat(1, k)
    row_idxs = row_idxs.long().view(-1).to(words.device)

    for t in range(max_words_in_sent):
        words = words.unsqueeze(1)

        # logprobs.shape=(batch, num_words)
        logprobs, kwargs = step_fn(words, t, **kwargs)

        if t == 0:
            topk_logprobs, topk_words = torch.topk(logprobs, k)
            # update
            words = topk_words.view(-1)
            logprobs = topk_logprobs.view(-1)
            step_words.append(words)
            step_word_logprobs.append(logprobs)
            step_prevs.append([])
            cum_logprob = logprobs
            if len(kwargs) > 0:
                kwargs = expand_fn(k, **kwargs)

        else:
            topk2_logprobs, topk2_words = torch.topk(logprobs, k)
            tmp_cum_logprob = topk2_logprobs + cum_logprob.unsqueeze(1)
            tmp_cum_logprob = tmp_cum_logprob.view(batch_size, k * k)
            topk2_words = topk2_words.view(batch_size, k * k)
            topk_cum_logprobs, topk_argwords = torch.topk(tmp_cum_logprob, k)
            topk_words = torch.gather(topk2_words, 1, topk_argwords)
            # update
            words = topk_words.view(-1)
            step_words.append(words)
            step_word_logprobs.append(torch.gather(
                topk2_logprobs.view(batch_size, k * k), 1, topk_argwords).view(-1))
            cum_logprob = topk_cum_logprobs.view(-1)

            # select previous hidden
            # prev_idxs.size = (batch, k)
            prev_idxs = topk_argwords.div(k).long().view(-1) + row_idxs
            step_prevs.append(prev_idxs)
            kwargs = select_fn(prev_idxs, **kwargs)
            finished_idxs = (words == EOS)

            for i, finished in enumerate(finished_idxs):
                b = i // k
                if batch_sent_pool_remain_cnt[b] > 0:
                    if finished or t == max_words_in_sent - 1:
                        batch_sent_pool_remain_cnt[b] -= 1

                        cmpl_sent, cmpl_word_logprobs = beam_search_recover_one_caption(
                            step_words, step_prevs, step_word_logprobs,
                            t, i, beam_width=beam_width)
                        sent_logprob = cum_logprob[i] / (t + 1)

                        sent_pool[b].append((sent_logprob, cmpl_sent, cmpl_word_logprobs))

            cum_logprob.masked_fill_(finished_idxs, -1000000)  # stop select the beam
            if np.sum(batch_sent_pool_remain_cnt) <= 0:
                break

    for i, sents in enumerate(sent_pool):
        sents.sort(key=lambda x: -x[0])
    return sent_pool


def sample_decode(words, step_fn, max_words_in_sent,
                  greedy=False, sample_topk=0, early_stop=True, **kwargs):
    """
    Args:
      words: init words, shape=(batch, )
      step_fn: function return word logprobs
      max_words_in_sent: int, max decoded sentence length
      greedy: greedy or multinomial sampling
      sample_topk: each step sample from topk words instead of all words
      early_stop: stop if all examples are ended
      kwargs for RNN decoders:
        - states: init decoder states, shape=(num_layers, batch, hidden_size)
        - outs: the last hidden layer (query in attn), shape=(batch, hidden_size*bi)
        - memory_keys: (optional for attn)
        - memory_values: (optional for attn)
        - memory_masks: (optional for attn)
      kwargs for Transformer decoders:
        -

    Returns:
      seq_words: int sent, LongTensor, shape=(batch, dec_seq_len)
      seq_word_logprobs: logprobs of the selected word, shape=(batch, dec_seq_len)
    """
    seq_words, seq_word_logprobs = [], []

    words = torch.unsqueeze(words, 1)
    unfinished = torch.ones_like(words).byte()

    for t in range(max_words_in_sent):
        logprobs, kwargs = step_fn(words, t, **kwargs)
        if greedy:
            _, words = torch.topk(logprobs, 1)
        else:
            probs = torch.exp(logprobs)
            if sample_topk > 0:
                topk_probs, topk_words = torch.topk(probs, sample_topk)
                idxs = torch.multinomial(topk_probs, 1)
                words = torch.gather(topk_words, 1, idxs)
            else:
                words = torch.multinomial(probs, 1)
        # words.shape=(batch, 1)
        seq_words.append(words)
        # logprobs.shape=(batch, num_words)
        logprobs = torch.gather(logprobs, 1, words)
        seq_word_logprobs.append(logprobs)
        unfinished = unfinished * (words != EOS)
        if early_stop and unfinished.sum().data.item() == 0:
            break
    seq_words = torch.cat(seq_words, 1).data
    seq_word_logprobs = torch.cat(seq_word_logprobs, 1)

    return seq_words, seq_word_logprobs


# class AbsModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = None
#         self.decoder = None
