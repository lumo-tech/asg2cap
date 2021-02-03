from torchvision.transforms import ToTensor
import numpy as np
import json
import os

word2int_file = os.path.join(os.path.dirname(__file__), 'word2int.json')
word2int = json.load(open(word2int_file))  # type:dict
int2word = {i: w for w, i in word2int.items()}

BOS = word2int.get('<BOS>', 0)
EOS = word2int.get('<EOS>', 1)
UNK = word2int.get('<UNK>', 2)


class Word2Int:
    def __call__(self, item):
        pass


class Int2Word:
    def __call__(self, item):
        pass


class Sent2Int:
    def __call__(self, str_sent):
        int_sent = [word2int.get(w, UNK) for w in str_sent.split()]
        return int_sent


class Int2Sent:
    def __call__(self, int_sent):
        str_sent = []
        for x in int_sent:
            if x == EOS:
                break
            str_sent.append(int2word[x])
        return ' '.join(str_sent)


class FitSentIDLen:
    def __init__(self, add_bos_eos=True,
                 max_words_in_sent=25):
        self.max_words_in_sent = max_words_in_sent
        self.add_bos_eos = add_bos_eos

    def __call__(self, int_sent):
        if self.add_bos_eos:
            sent = [BOS] + int_sent + [EOS]
        else:
            sent = int_sent
        sent = sent[:self.max_words_in_sent]
        num_pad = self.max_words_in_sent - len(sent)
        mask = [True] * len(sent) + [False] * num_pad
        sent = sent + [EOS] * num_pad
        return sent, mask


class FitFeatureLen:
    def __init__(self, max_len, average=False):
        self.max_len = max_len
        self.average = average

    def __call__(self, attn_ft):
        max_len, average = self.max_len, self.average

        seq_len, dim_ft = attn_ft.shape
        mask = np.zeros((max_len,), np.bool)

        # pad
        if seq_len < max_len:
            new_ft = np.zeros((max_len, dim_ft), np.float32)
            new_ft[:seq_len] = attn_ft
            mask[:seq_len] = True
        elif seq_len == max_len:
            new_ft = attn_ft
            mask[:] = True
        # trim
        else:
            if average:
                idxs = np.round(np.linspace(0, seq_len, max_len + 1)).astype(np.long)
                new_ft = np.array([np.mean(attn_ft[idxs[i]: idxs[i + 1]], axis=0) for i in range(max_len)])
            else:
                idxs = np.round(np.linspace(0, seq_len - 1, max_len)).astype(np.long)
                new_ft = attn_ft[idxs]
            mask[:] = True
        return np.array(new_ft), mask
