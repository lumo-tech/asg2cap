from thexp import Trainer
from thexp.frame import callbacks

from .. import GlobalParams


class BaseCBMixin(Trainer):
    def callbacks(self, params: GlobalParams):
        callbacks.LoggerCallback().hook(self)  # auto log in screen and file
        callbacks.EvalCallback(5, 10).hook(self)  # auto eval/test per 5/10 epoch
        callbacks.AutoRecord().hook(self)  # auto record meter by SummaryWritter

        callbacks.TimingCheckpoint(10).hook(self)

        # callbacks.LRSchedule().hook(self)  # auto get params.lr_sche to apply lr rate
        if params.ema:
            callbacks.EMAUpdate().hook(self)  # auto update module named with prefix `ema`

        if isinstance(self, callbacks.TrainCallback):
            self.hook(self)
