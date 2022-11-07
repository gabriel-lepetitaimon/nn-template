from .. import Cfg


@Cfg.register_obj("hardware")
class HardwareCfg(Cfg.Obj):
    debug = False
    gpus = Cfg.str(None)
    num_worker = 0

    minibatch_splits = 1
    precision = Cfg.oneOf('64', '32', 'bf16', '16')
    cudnn_benchmark = False

    @gpus.checker
    def check_gpus(self, v):
        if ',' in v:
            v = [int(_.strip()) for _ in v.split(',') if _.strip()]
            return v
        return int(v)

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

