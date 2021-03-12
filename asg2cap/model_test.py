from trainers import GlobalParams

params = GlobalParams()

from thexp import Trainer

from trainers.mixin.models import AsgModelMixin

trainer = Trainer(params)
AsgModelMixin.models(trainer, params)

print((trainer.model))
