import torch
from torch import nn
from torch.nn import functional as F

from arch.attentions import GlobalAttention
from arch.embedding import Embedding


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = Embedding(self.config.num_words,
                                   self.config.dim_word, fix_word_embed=self.config.fix_word_embed)

        kwargs = {}
        if self.config.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.config.rnn_type,
                               input_size=self.rnn_input_size, hidden_size=self.config.hidden_size,
                               num_layers=self.config.num_layers, dropout=self.config.dropout,
                               bias=True, batch_first=True, **kwargs)
        else:
            assert False

        if self.config.hidden2word:
            self.hidden2word = nn.Linear(self.config.hidden_size, self.config.dim_word)
            output_size = self.config.dim_word
        else:
            output_size = self.config.hidden_size

        if not self.config.tie_embed:
            self.fc = nn.Linear(output_size, self.config.num_words)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(self.config.dropout)

        self.init_rnn_weights(self.rnn, self.config.rnn_type)

    @property
    def rnn_input_size(self):
        return self.config.dim_word

    # TODO rm
    def init_rnn_weights(self, rnn, rnn_type, num_layers=None):
        if rnn_type == 'lstm':
            # the ordering of weights a biases is ingate, forgetgate, cellgate, outgate
            # init forgetgate as 1 to make rnn remember the past in the beginning
            if num_layers is None:
                num_layers = rnn.num_layers
            for layer in range(num_layers):
                for name in ['i', 'h']:
                    try:
                        weight = getattr(rnn, 'weight_%sh_l%d' % (name, layer))
                    except:
                        weight = getattr(rnn, 'weight_%sh' % name)
                    nn.init.orthogonal_(weight.data)
                    try:
                        bias = getattr(rnn, 'bias_%sh_l%d' % (name, layer))
                    except:
                        bias = getattr(rnn, 'bias_%sh' % name)  # BUTD: LSTM Cell
                    nn.init.constant_(bias, 0)
                    if name == 'i':
                        bias.data.index_fill_(0, torch.arange(
                            rnn.hidden_size, rnn.hidden_size * 2).long(), 1)
                        # bias.requires_grad = False

    def init_dec_state(self, encoder_state):
        '''
          The encoder hidden is (batch, dim_embed)
          We need to convert it to (layers, batch, hidden_size)
          assert dim_embed == hidden_size
        '''
        decoder_state = encoder_state.repeat(self.config.num_layers, 1, 1)
        if self.config.rnn_type == 'lstm' or self.config.rnn_type == 'ONLSTM':
            decoder_state = tuple([decoder_state, decoder_state])
        return decoder_state

    def calc_logits_with_rnn_outs(self, outs):
        '''
        Args:
          outs: (batch, hidden_size)
        Returns:
          logits: (batch, num_words)
        '''
        if self.config.hidden2word:
            outs = torch.tanh(self.hidden2word(outs))
        outs = self.dropout(outs)
        if self.config.tie_embed:
            logits = torch.mm(outs, self.embedding.we.weight.t())
        else:
            logits = self.fc(outs)
        return logits

    def forward(self, inputs, encoder_state):
        '''
        Args:
          inputs: (batch, seq_len)
          encoder_state: (batch, dim_embed)
        Returns:
          logits: (batch*seq_len, num_words)
        '''
        states = self.init_dec_state(encoder_state)

        if self.config.schedule_sampling:
            step_outs = []
            for t in range(inputs.size(1)):
                wordids = inputs[:, t]
                if t > 0:
                    sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
                    sample_mask = sample_rate < self.config.ss_rate
                    prob = self.softmax(step_outs[-1]).detach()
                    sampled_wordids = torch.multinomial(prob, 1).squeeze(1)
                    wordids.masked_scatter_(sample_mask, sampled_wordids)
                embed = self.embedding(wordids)
                embed = self.dropout(embed)
                outs, states = self.rnn(embed.unsqueeze(1), states)
                outs = outs[:, 0]
                logit = self.calc_logits_with_rnn_outs(outs)
                step_outs.append(logit)
            logits = torch.stack(step_outs, 1)
            logits = logits.view(-1, self.config.num_words)
        # pytorch rnn utilzes cudnn to speed up
        else:
            embeds = self.embedding(inputs)
            embeds = self.dropout(embeds)
            # outs.size(batch, seq_len, hidden_size)
            outs, states = self.rnn(embeds, states)
            outs = outs.contiguous().view(-1, self.config.hidden_size)
            logits = self.calc_logits_with_rnn_outs(outs)
        return logits


class AttnDecoder(Decoder):
    def __init__(self, config):
        super().__init__(config)

        self.attn = GlobalAttention(self.config.hidden_size, self.config.attn_size, self.config.attn_type)
        if self.config.attn_type == 'mlp':
            self.attn_linear_context = nn.Linear(self.config.attn_input_size,
                                                 self.config.attn_size, bias=False)

        if not self.config.memory_same_key_value:
            self.memory_value_layer = nn.Linear(self.config.attn_input_size,
                                                self.config.attn_size, bias=True)

    @property
    def rnn_input_size(self):
        if self.config.rnn_input_size:
            return self.config.dim_word + self.config.attn_input_size
        else:
            return self.config.dim_word + self.config.attn_size

    def gen_memory_key_value(self, enc_memories):
        if self.config.memory_same_key_value:
            memory_values = enc_memories
        else:
            memory_values = F.relu(self.memory_value_layer(enc_memories))

        if self.config.attn_type == 'mlp':
            memory_keys = self.attn_linear_context(enc_memories)
        else:
            memory_keys = enc_memories

        return memory_keys, memory_values

    def forward(self, inputs, enc_states, enc_memories, enc_masks, return_attn=False):
        '''
        Args:
          inputs: (batch, dec_seq_len)
          enc_states: (batch, dim_embed)
          enc_memoris: (batch, enc_seq_len, dim_embed)
          enc_masks: (batch, enc_seq_len)
        Returns:
          logits: (batch*seq_len, num_words)
        '''
        memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
        states = self.init_dec_state(enc_states)
        outs = states[0][-1] if isinstance(states, tuple) else states[-1]

        step_outs, step_attns = [], []
        for t in range(inputs.size(1)):
            wordids = inputs[:, t]
            if t > 0 and self.config.schedule_sampling:
                sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
                sample_mask = sample_rate < self.config.ss_rate
                prob = self.softmax(step_outs[-1]).detach()
                sampled_wordids = torch.multinomial(prob, 1).view(-1)
                wordids.masked_scatter_(sample_mask, sampled_wordids)
            embed = self.embedding(wordids)
            attn_score, attn_memory = self.attn(outs,
                                                memory_keys, memory_values, enc_masks)
            step_attns.append(attn_score)
            rnn_input = torch.cat([embed, attn_memory], 1).unsqueeze(1)
            rnn_input = self.dropout(rnn_input)
            outs, states = self.rnn(rnn_input, states)
            outs = outs[:, 0]
            logit = self.calc_logits_with_rnn_outs(outs)
            step_outs.append(logit)

        logits = torch.stack(step_outs, 1)
        logits = logits.view(-1, self.config.num_words)

        if return_attn:
            return logits, step_attns
        return logits


class BUTDAttnDecoder(AttnDecoder):
    '''
    Requires: dim input visual feature == lstm hidden size
    '''

    def __init__(self, config):
        nn.Module.__init__(self)  # need to rewrite RNN
        self.config = config
        # word embedding
        self.embedding = Embedding(self.config.num_words,
                                   self.config.dim_word, fix_word_embed=self.config.fix_word_embed)
        # rnn params (attn_lstm and lang_lstm)
        self.attn_lstm = nn.LSTMCell(
            self.config.hidden_size + self.config.attn_input_size + self.config.dim_word,  # (h_lang, v_g, w)
            self.config.hidden_size, bias=True)
        memory_size = self.config.attn_input_size if self.config.memory_same_key_value else self.config.attn_size
        self.lang_lstm = nn.LSTMCell(
            self.config.hidden_size + memory_size,  # (h_attn, v_a)
            self.config.hidden_size, bias=True)
        # attentions
        self.attn = GlobalAttention(self.config.hidden_size, self.config.attn_size, self.config.attn_type)
        if self.config.attn_type == 'mlp':
            self.attn_linear_context = nn.Linear(self.config.attn_input_size, self.config.attn_size, bias=False)
        if not self.config.memory_same_key_value:
            self.memory_value_layer = nn.Linear(self.config.attn_input_size, self.config.attn_size, bias=True)
        # outputs
        if self.config.hidden2word:
            self.hidden2word = nn.Linear(self.config.hidden_size, self.config.dim_word)
            output_size = self.config.dim_word
        else:
            output_size = self.config.hidden_size
        if not self.config.tie_embed:
            self.fc = nn.Linear(output_size, self.config.num_words)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(self.config.dropout)
        self.init_rnn_weights(self.attn_lstm, 'lstm', num_layers=1)
        self.init_rnn_weights(self.lang_lstm, 'lstm', num_layers=1)

    def init_dec_state(self, batch_size):
        param = next(self.parameters())
        states = []
        for i in range(2):  # (hidden, cell)
            states.append(torch.zeros((2, batch_size, self.config.hidden_size),
                                      dtype=torch.float32).to(param.device))
        return states

    def forward(self, inputs, enc_globals, enc_memories, enc_masks, return_attn=False):
        '''
        Args:
          inputs: (batch, dec_seq_len)
          enc_globals: (batch, hidden_size)
          enc_memories: (batch, enc_seq_len, attn_input_size)
          enc_masks: (batch, enc_seq_len)
        Returns:
          logits: (batch*seq_len, num_words)
        '''
        memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
        states = self.init_dec_state(inputs.size(0))  # zero init state

        step_outs, step_attns = [], []
        for t in range(inputs.size(1)):
            wordids = inputs[:, t]
            if t > 0 and self.config.schedule_sampling:
                sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
                sample_mask = sample_rate < self.config.ss_rate
                prob = self.softmax(step_outs[-1]).detach()  # detach grad
                sampled_wordids = torch.multinomial(prob, 1).view(-1)
                wordids.masked_scatter_(sample_mask, sampled_wordids)
            embed = self.embedding(wordids)

            h_attn_lstm, c_attn_lstm = self.attn_lstm(
                torch.cat([states[0][1], enc_globals, embed], dim=1),
                (states[0][0], states[1][0]))

            attn_score, attn_memory = self.attn(h_attn_lstm,
                                                memory_keys, memory_values, enc_masks)
            step_attns.append(attn_score)

            h_lang_lstm, c_lang_lstm = self.lang_lstm(
                torch.cat([h_attn_lstm, attn_memory], dim=1),
                (states[0][1], states[1][1]))

            outs = h_lang_lstm
            logit = self.calc_logits_with_rnn_outs(outs)
            step_outs.append(logit)
            states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
                      torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

        logits = torch.stack(step_outs, 1)
        logits = logits.view(-1, self.config.num_words)

        if return_attn:
            return logits, step_attns
        return logits


class MemoryDecoder(BUTDAttnDecoder):
    def __init__(self, config):
        super().__init__(config)

        memory_size = self.config.attn_size if self.config.memory_same_key_value else self.config.attn_input_size
        self.memory_update_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size + memory_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, memory_size * 2))
        self.sentinal_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1))

    def forward(self, inputs, enc_globals, enc_memories, enc_masks, return_attn=False):
        '''
        Args:
          inputs: (batch, dec_seq_len)
          enc_globals: (batch, hidden_size)
          enc_memories: (batch, enc_seq_len, attn_input_size)
          enc_masks: (batch, enc_seq_len)
        Returns:
          logits: (batch*seq_len, num_words)
        '''
        enc_seq_len = enc_memories.size(1)
        memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
        states = self.init_dec_state(inputs.size(0))  # zero init state

        step_outs, step_attns = [], []
        for t in range(inputs.size(1)):
            wordids = inputs[:, t]
            if t > 0 and self.config.schedule_sampling:
                sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
                sample_mask = sample_rate < self.config.ss_rate
                prob = self.softmax(step_outs[-1]).detach()  # detach grad
                sampled_wordids = torch.multinomial(prob, 1).view(-1)
                wordids.masked_scatter_(sample_mask, sampled_wordids)
            embed = self.embedding(wordids)

            h_attn_lstm, c_attn_lstm = self.attn_lstm(
                torch.cat([states[0][1], enc_globals, embed], dim=1),
                (states[0][0], states[1][0]))

            # attn_score: (batch, max_attn_len)
            attn_score, attn_memory = self.attn(h_attn_lstm,
                                                memory_keys, memory_values, enc_masks)
            step_attns.append(attn_score)

            h_lang_lstm, c_lang_lstm = self.lang_lstm(
                torch.cat([h_attn_lstm, attn_memory], dim=1),
                (states[0][1], states[1][1]))

            # write: update memory keys and values
            # (batch, enc_seq_len, hidden_size + attn_input_size)
            individual_vectors = torch.cat(
                [h_lang_lstm.unsqueeze(1).expand(-1, enc_seq_len, -1), enc_memories], 2)
            update_vectors = self.memory_update_layer(individual_vectors)
            memory_size = update_vectors.size(-1) // 2
            erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
            add_vectors = update_vectors[:, :, memory_size:]

            # some words do not need to attend on visual nodes
            sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
            memory_attn_score = attn_score * sentinal_gates

            enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                           + memory_attn_score.unsqueeze(2) * add_vectors
            memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

            outs = h_lang_lstm
            logit = self.calc_logits_with_rnn_outs(outs)
            step_outs.append(logit)
            states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
                      torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

        logits = torch.stack(step_outs, 1)
        logits = logits.view(-1, self.config.num_words)

        if return_attn:
            return logits, step_attns
        return logits


class ContentFlowAttentionDecoder(BUTDAttnDecoder):
    def __init__(self, config):
        super().__init__(config)

        memory_size = self.config.attn_size if self.config.memory_same_key_value else self.config.attn_input_size
        self.address_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size + memory_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, 1 + 3))

    def forward(self, inputs, enc_globals, enc_memories, enc_masks, flow_edges, return_attn=False):
        '''
        Args:
          inputs: (batch, dec_seq_len)
          enc_globals: (batch, hidden_size)
          enc_memories: (batch, enc_seq_len, attn_input_size)
          enc_masks: (batch, enc_seq_len)
        Returns:
          logits: (batch*seq_len, num_words)
        '''
        batch_size, max_attn_len = enc_masks.size()
        device = inputs.device

        states = self.init_dec_state(batch_size)  # zero init state

        # initialize content attention
        memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

        # initialize location attention score: (batch, max_attn_len)
        prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device)
        prev_attn_score[:, 0] = 1

        step_outs, step_attns = [], []
        for t in range(inputs.size(1)):
            wordids = inputs[:, t]
            if t > 0 and self.config.schedule_sampling:
                sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
                sample_mask = sample_rate < self.config.ss_rate
                prob = self.softmax(step_outs[-1]).detach()  # detach grad
                sampled_wordids = torch.multinomial(prob, 1).view(-1)
                wordids.masked_scatter_(sample_mask, sampled_wordids)
            embed = self.embedding(wordids)

            h_attn_lstm, c_attn_lstm = self.attn_lstm(
                torch.cat([states[0][1], enc_globals, embed], dim=1),
                (states[0][0], states[1][0]))

            prev_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
            address_params = self.address_layer(torch.cat([h_attn_lstm, prev_memory], 1))
            interpolate_gate = torch.sigmoid(address_params[:, :1])
            flow_gate = torch.softmax(address_params[:, 1:], dim=1)

            # content_attn_score: (batch, max_attn_len)
            content_attn_score, content_attn_memory = self.attn(h_attn_lstm,
                                                                memory_keys, memory_values, enc_masks)

            # location attention flow: (batch, max_attn_len)
            flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prev_attn_score)
            flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
            # (batch, max_attn_len, 3)
            flow_attn_score = torch.stack([x.view(batch_size, max_attn_len) \
                                           for x in [prev_attn_score, flow_attn_score_1, flow_attn_score_2]], 2)
            flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow_attn_score, 2)

            # content + location interpolation
            attn_score = interpolate_gate * content_attn_score + (1 - interpolate_gate) * flow_attn_score

            # final attention
            step_attns.append(attn_score)
            prev_attn_score = attn_score
            attn_memory = torch.sum(attn_score.unsqueeze(2) * memory_values, 1)

            # next layer with attended context
            h_lang_lstm, c_lang_lstm = self.lang_lstm(
                torch.cat([h_attn_lstm, attn_memory], dim=1),
                (states[0][1], states[1][1]))

            outs = h_lang_lstm
            logit = self.calc_logits_with_rnn_outs(outs)
            step_outs.append(logit)
            states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
                      torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

        logits = torch.stack(step_outs, 1)
        logits = logits.view(-1, self.config.num_words)

        if return_attn:
            return logits, step_attns
        return logits


class MemoryFlowDecoder(ContentFlowAttentionDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        memory_size = self.config.attn_size if self.config.memory_same_key_value else self.config.attn_input_size
        self.memory_update_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size + memory_size, memory_size),
            nn.ReLU(),
            nn.Linear(memory_size, memory_size * 2))
        self.sentinal_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1))

    def forward(self, inputs, enc_globals, enc_memories, enc_masks, flow_edges, return_attn=False):
        '''
        Args:
          inputs: (batch, dec_seq_len)
          enc_globals: (batch, hidden_size)
          enc_memories: (batch, enc_seq_len, attn_input_size)
          enc_masks: (batch, enc_seq_len)
          flow_edges: sparse matrix (num_nodes, num_nodes), num_nodes=batch*enc_seq_len
        Returns:
          logits: (batch*seq_len, num_words)
        '''
        batch_size, max_attn_len = enc_masks.size()
        device = inputs.device

        # initialize states
        states = self.init_dec_state(batch_size)  # zero init state

        # location attention: (batch, max_attn_len)
        prev_attn_score = torch.zeros((batch_size, max_attn_len)).to(device)
        prev_attn_score[:, 0] = 1

        step_outs, step_attns = [], []
        for t in range(inputs.size(1)):
            wordids = inputs[:, t]
            if t > 0 and self.config.schedule_sampling:
                sample_rate = torch.rand(wordids.size(0)).to(wordids.device)
                sample_mask = sample_rate < self.config.ss_rate
                prob = self.softmax(step_outs[-1]).detach()  # detach grad
                sampled_wordids = torch.multinomial(prob, 1).view(-1)
                wordids.masked_scatter_(sample_mask, sampled_wordids)
            embed = self.embedding(wordids)

            h_attn_lstm, c_attn_lstm = self.attn_lstm(
                torch.cat([states[0][1], enc_globals, embed], dim=1),
                (states[0][0], states[1][0]))

            memory_keys, memory_values = self.gen_memory_key_value(enc_memories)

            prev_attn_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
            address_params = self.address_layer(torch.cat([h_attn_lstm, prev_attn_memory], 1))
            interpolate_gate = torch.sigmoid(address_params[:, :1])
            flow_gate = torch.softmax(address_params[:, 1:], dim=1)

            # content_attn_score: (batch, max_attn_len)
            content_attn_score, content_attn_memory = self.attn(h_attn_lstm,
                                                                memory_keys, memory_values, enc_masks)

            # location attention flow: (batch, max_attn_len)
            flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prev_attn_score)
            flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
            # (batch, max_attn_len, 3)
            flow_attn_score = torch.stack([x.view(batch_size, max_attn_len) \
                                           for x in [prev_attn_score, flow_attn_score_1, flow_attn_score_2]], 2)
            flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow_attn_score, 2)

            # content + location interpolation
            attn_score = interpolate_gate * content_attn_score + (1 - interpolate_gate) * flow_attn_score

            # final attention
            step_attns.append(attn_score)
            prev_attn_score = attn_score
            attn_memory = torch.sum(attn_score.unsqueeze(2) * memory_values, 1)

            # next layer with attended context
            h_lang_lstm, c_lang_lstm = self.lang_lstm(
                torch.cat([h_attn_lstm, attn_memory], dim=1),
                (states[0][1], states[1][1]))

            # write: update memory keys and values
            individual_vectors = torch.cat([h_lang_lstm.unsqueeze(1).expand(-1, max_attn_len, -1), enc_memories], 2)
            update_vectors = self.memory_update_layer(individual_vectors)
            memory_size = update_vectors.size(-1) // 2
            erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
            add_vectors = update_vectors[:, :, memory_size:]

            # some words do not need to attend on visual nodes
            sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
            memory_attn_score = attn_score * sentinal_gates

            enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                           + memory_attn_score.unsqueeze(2) * add_vectors

            outs = h_lang_lstm
            logit = self.calc_logits_with_rnn_outs(outs)
            step_outs.append(logit)
            states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
                      torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

        logits = torch.stack(step_outs, 1)
        logits = logits.view(-1, self.config.num_words)

        if return_attn:
            return logits, step_attns
        return logits

    def step_fn(self, words, step, **kwargs):
        states = kwargs['states']
        enc_globals = kwargs['enc_globals']
        enc_memories = kwargs['enc_memories']
        memory_masks = kwargs['memory_masks']
        prev_attn_score = kwargs['prev_attn_score']
        flow_edges = kwargs['flow_edges']

        batch_size, max_attn_len = memory_masks.size()
        memory_keys, memory_values = self.gen_memory_key_value(enc_memories)
        embed = self.embedding(words.squeeze(1))

        h_attn_lstm, c_attn_lstm = self.attn_lstm(
            torch.cat([states[0][1], enc_globals, embed], dim=1),
            (states[0][0], states[1][0]))

        prev_attn_memory = torch.sum(prev_attn_score.unsqueeze(2) * memory_values, 1)
        address_params = self.address_layer(torch.cat([h_attn_lstm, prev_attn_memory], 1))
        interpolate_gate = torch.sigmoid(address_params[:, :1])
        flow_gate = torch.softmax(address_params[:, 1:], dim=1)

        # content_attn_score: (batch, max_attn_len)
        content_attn_score, content_attn_memory = self.attn(h_attn_lstm,
                                                            memory_keys, memory_values, memory_masks)

        # location attention flow: (batch, max_attn_len)
        flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prev_attn_score)
        flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
        flow_attn_score = torch.stack([x.view(batch_size, max_attn_len) \
                                       for x in [prev_attn_score, flow_attn_score_1, flow_attn_score_2]], 2)
        flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow_attn_score, 2)

        # content + location interpolation
        attn_score = interpolate_gate * content_attn_score + (1 - interpolate_gate) * flow_attn_score

        # final attention
        attn_memory = torch.sum(attn_score.unsqueeze(2) * memory_values, 1)

        h_lang_lstm, c_lang_lstm = self.lang_lstm(
            torch.cat([h_attn_lstm, attn_memory], dim=1),
            (states[0][1], states[1][1]))

        logits = self.calc_logits_with_rnn_outs(h_lang_lstm)
        logprobs = self.log_softmax(logits)
        states = (torch.stack([h_attn_lstm, h_lang_lstm], dim=0),
                  torch.stack([c_attn_lstm, c_lang_lstm], dim=0))

        # write: update memory keys and values
        individual_vectors = torch.cat([h_lang_lstm.unsqueeze(1).expand(-1, max_attn_len, -1), enc_memories], 2)
        update_vectors = self.memory_update_layer(individual_vectors)
        memory_size = update_vectors.size(-1) // 2
        erase_gates = torch.sigmoid(update_vectors[:, :, :memory_size])
        add_vectors = update_vectors[:, :, memory_size:]

        sentinal_gates = torch.sigmoid(self.sentinal_layer(h_lang_lstm))
        memory_attn_score = attn_score * sentinal_gates
        enc_memories = enc_memories * (1 - memory_attn_score.unsqueeze(2) * erase_gates) \
                       + memory_attn_score.unsqueeze(2) * add_vectors

        kwargs['states'] = states
        kwargs['enc_memories'] = enc_memories
        kwargs['prev_attn_score'] = attn_score
        return logprobs, kwargs
