import os
from . import Cfg


@Cfg.register_obj("hardware")
class HardwareCfg(Cfg.Obj):
    debug = Cfg.oneOf(False, True, 'fast', default=False)
    gpus = Cfg.str(None)
    num_workers = Cfg.oneOf('auto', Cfg.int(min=0), default='auto')

    minibatch_splits = Cfg.int(1, min=1)
    precision = Cfg.oneOf('64', '32', 'bf16', '16', default=None)
    cudnn_benchmark = Cfg.bool(False)

    @gpus.checker
    def check_gpus(self, v):
        if ',' in v:
            v = [int(_.strip()) for _ in v.split(',') if _.strip()]
            return v
        return int(v)

    @num_workers.post_checker
    def check_num_workers(self, v):
        cpus = os.cpu_count()
        match v:
            case 'auto':
                return cpus
            case int():
                return min(cpus, v)

    def lightning_args(self):
        args = dict(fast_dev_run=self.debug,
                    enable_progress_bar=self.debug,
                    accumulate_grad_batches=self.minibatch_splits)

        if self.gpus is None:
            args.update(dict(
                accelerator='gpus',
                devices=self.gpus,
                auto_select_gpus=isinstance(self.gpus, int),
                benchmark=self.cudnn_benchmark,
                             ))
        return args

