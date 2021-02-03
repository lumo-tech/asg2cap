from torch.nn import functional as F

from arch.base import *


class GlobalAttention(nn.Module):
    '''
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`,
    to an output of size `dim`.

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.

    However they differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`

    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = w_a^T tanh(W_a q + U_a h_j)`

    Args:
       attn_size (int): dimensionality of query and key
       attn_type (str): type of attention to use, options [dot,general,mlp]
    '''

    def __init__(self, query_size, attn_size, attn_type='dot'):
        super(GlobalAttention, self).__init__()

        self.query_size = query_size
        self.attn_size = attn_size
        self.attn_type = attn_type

        if self.attn_type == 'general':
            self.linear_in = nn.Linear(query_size, attn_size, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_query = nn.Linear(query_size, attn_size, bias=True)
            self.attn_w = nn.Linear(attn_size, 1, bias=False)
        elif self.attn_type == 'dot':
            assert self.query_size == self.attn_size

    def forward(self, query, memory_keys, memory_values, memory_masks):
        """
        Args:
          query (`FloatTensor`): (batch, query_size)
          memory_keys (`FloatTensor`): (batch, seq_len, attn_size)
          memory_values (`FloatTensor`): (batch, seq_len, attn_size)
          memory_masks (`LongTensor`): (batch, seq_len)

        Returns:
          attn_score: attention distributions (batch, seq_len)
          attn_memory: computed context vector, (batch, attn_size)
        """
        batch_size, seq_len, attn_size = memory_keys.size()

        if self.attn_type == 'mlp':
            query_hidden = self.linear_query(query.unsqueeze(1)).expand(
                batch_size, seq_len, attn_size)
            # attn_hidden: # (batch, seq_len, attn_size)
            attn_hidden = torch.tanh(query_hidden + memory_keys)
            # attn_score: (batch, seq_len, 1)
            attn_score = self.attn_w(attn_hidden)
        elif self.attn_type == 'dot':
            # attn_score: (batch, seq_len, 1)
            attn_score = torch.bmm(memory_keys, query.unsqueeze(2))
        elif self.attn_type == 'general':
            query_hidden = self.linear_in(query)
            attn_score = torch.bmm(memory_keys, query_hidden.unsqueeze(2))
        else:
            assert False

        # attn_score: (batch, seq_len)
        attn_score = attn_score.squeeze(2)
        if memory_masks is not None:
            attn_score = attn_score * memory_masks  # memory mask [0, 1]
            attn_score = attn_score.masked_fill(memory_masks == 0, -1e18)
        attn_score = F.softmax(attn_score, dim=1)
        # make sure no item is attended when all memory_masks are all zeros
        if memory_masks is not None:
            attn_score = attn_score.masked_fill(memory_masks == 0, 0)
        attn_memory = torch.sum(attn_score.unsqueeze(2) * memory_values, 1)
        return attn_score, attn_memory
