from .config.config import Cfg


@Cfg.register_obj('experiment')
class ExperimentCfg(Cfg.Obj):
    name = Cfg.str()
    run_id = Cfg.int(None)
    project = Cfg.str(None)
    group = Cfg.str(None)
    job_type = Cfg.str(None)
    tags = Cfg.strList([])
    dir = Cfg.str(None)
    notes = Cfg.str(None)

    @property
    def run_name(self):
        if self.run_id is not None:
            return self.name + f"-{self.run_id:02d}"
        return self.name

    def init_wandb(self):
        import wandb
        wandb.init(
            name=self.run_name,
            project=self.project,
            job_type=self.job_type,
            tags=self.tags,
            config=self.root().to_dict(exportable=True),
            dir=self.dir,
            notes=self.notes
        )

    