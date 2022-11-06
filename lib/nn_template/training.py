from . import Cfg


@Cfg.register_obj("training")
class TrainingCfg(Cfg.Obj):
    seed = 1234
    minibatch = Cfg.int()
    max_epoch = Cfg.int()


