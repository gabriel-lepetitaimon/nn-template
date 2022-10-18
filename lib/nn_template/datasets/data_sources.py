from ..config import Cfg

class DataSource(Cfg.Obj):
    def __init__(self):
        pass


class DataSourcesAttr(Cfg.multi_type_collection):
    def __init__(self):
        super(DataSourcesAttr, self).__init__(
                type_key='type',
                obj_types={'Images': ..., })



class