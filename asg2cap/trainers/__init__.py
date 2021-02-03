from thexp import Params


class GlobalParams(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.optim = self.create_optim('Adam',
                                       lr=2e-4,
                                       weight_decay=0)

        self.architecture = self.choice('architecture',
                                        'rgcn.flow.memory',
                                        'node', 'node.role',
                                        'rgcn', 'rgcn.flow', 'rgcn.memory')

        self.attn_encoder = self.choice('attn_encoder',
                                        'base', 'flat',
                                        'rgcn', 'role_rgcn')

        self.decoder = self.choice('decoder',
                                   'base',
                                   'memory', 'memory_flow',
                                   'cont_flow_attn'
                                   'attn', 'butd_attn')

        self.dataset = self.choice('dataset', 'mscoco')
        self.n_classes = 10
        self.topk = (1, 2, 3, 4)

        self.batch_size = 10

        self.num_workers = 4

        self.ema = True
        self.ema_alpha = 0.999

        self.val_size = 10000

        self.max_attn_len = 10
        self.pixel_reduce = 1

        self.dim_hidden = 512
        self.dim_input = 2048
        self.dropout = 0.0
        self.embed_first = True
        self.freeze = False
        self.max_attn_len = 10
        self.num_hidden_layers = 2
        self.num_node_types = 3
        self.num_rels = 6
        self.self_loop = True
        self.weight_decay = 0
        self.attn_input_size = 512
        self.attn_size = 512
        self.attn_type = 'mlp'
        self.beam_width = 1
        self.dim_word = 512
        self.dropout = 0.5
        self.fix_word_embed = False
        self.freeze = False
        self.greedy_or_beam = False
        self.hidden2word = False
        self.hidden_size = 512
        self.max_words_in_sent = 25
        self.memory_same_key_value = True
        self.num_layers = 1
        self.num_words = 10942
        self.schedule_sampling = False
        self.sent_pool_size = 1
        self.ss_increase_epoch = 5
        self.ss_increase_rate = 0.05
        self.ss_max_rate = 0.25
        self.ss_rate = 0.0
        self.tie_embed = True
        self.weight_decay = 0
        self.dim_embed = 512
        self.dim_fts = [2048,512]
        self.dropout = 0
        self.freeze = False
        self.is_embed = True
        self.lr_mult = 1.0
        self.nonlinear = False
        self.norm = False

    def initial(self):

        if self.ENV.IS_PYCHARM_DEBUG:
            self.num_workers = 0

        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00001,
                                     right=self.epoch)

        # Auto Choice encoder
        if self.architecture == 'node':
            self.attn_encoder = 'base'
        elif self.architecture == 'node.role':
            self.attn_encoder = 'flat'
        elif self.architecture in {'rgcn',
                                   'rgcn.flow',
                                   'rgcn.memory', 'rgcn.flow.memory'}:
            self.attn_encoder = 'role_rgcn'

        # Auto Choice decoder
        if self.architecture in {'node', 'node.role', 'rgcn'}:
            self.decoder = 'butd_attn'
        elif self.architecture == 'rgcn.flow':
            self.decoder = 'cont_flow_attn'
        elif self.architecture == 'rgcn.memory':
            self.decoder = 'memory'
        elif self.architecture == 'rgcn.flow.memory':
            self.decoder = 'memory_flow'


class EncoderConfig(Params):
    def __init__(self):
        super().__init__()
        self.dim_fts = [2048]
        self.dim_embed = 512
        self.is_embed = True
        self.dropout = 0
        self.norm = False
        self.nonlinear = False

    def _assert(self):
        if not self.is_embed:
            assert self.dim_embed == sum(self.dim_fts)
