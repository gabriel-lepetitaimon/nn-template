import os.path as P
from torch.utils.data import DataLoader

from ..config import default_config
from .data_augment import parse_data_augmentations


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


def load_dataset(cfg=None):
    if cfg is None:
        cfg = default_config()

    batch_size = cfg['hyper-parameters']['batch-size']
    steered = cfg.get('model.steered', False)

    data_path = cfg.training.get('dataset-path', 'default')
    if data_path == 'default':
        data_path = DEFAULT_DATA_PATH

    if not isinstance(steered, str):
        steered = cfg.get('model.steered.steering', False)
    if steered == 'attention':
        steered = False

    if 'datasets' in cfg:
        data_augmentations = parse_data_augmentations(cfg)
        if cfg.datasets.type == 'GenericHDF':
            from .generic_hdf import create_generic_hdf_datasets
            train, val, test = create_generic_hdf_datasets(cfg.datasets, data_path, cfg.get('training.seed', 1234),
                                                           data_augmentations)
        else:
            raise ValueError(f'Invalid dataset type: cfg.dataset.type={cfg.dataset.type}')
        trainD = DataLoader(train, pin_memory=True, shuffle=True, batch_size=batch_size,
                            num_workers=cfg.training['num-worker'] )
        validD = DataLoader(val, pin_memory=True, num_workers=6, batch_size=6)
        testD = {k: DataLoader(v, pin_memory=True, num_workers=6, batch_size=6)
                 for k, v in test.items()}
        return trainD, validD, testD
    else:
        from .legacy import TrainDataset, TestDataset
        train_dataset = cfg.training['training-dataset']
        dataset_file = P.join(data_path, cfg.training['dataset-file'])
        cfg['data-augmentation']['seed'] = cfg.get('training/seed', 1234)
        trainD = DataLoader(TrainDataset('train/'+train_dataset, file=dataset_file,
                                         factor=cfg.training['training-dataset-factor'],
                                         steered=steered, use_preprocess=cfg.training['use-preprocess'],
                                         data_augmentation_cfg=cfg['data-augmentation']),
                            pin_memory=True, shuffle=True,
                            batch_size=batch_size,
                            num_workers=cfg.training['num-worker']
                            )
        validD = DataLoader(TestDataset('val/'+train_dataset, file=dataset_file, steered=steered,
                                        use_preprocess=cfg.training['use-preprocess'],),
                            pin_memory=True, num_workers=6, batch_size=6)
        testD = {_: DataLoader(TestDataset('test/'+_, file=dataset_file, steered=steered,
                                           use_preprocess=cfg.training['use-preprocess'],),
                               pin_memory=True, num_workers=6, batch_size=6)
                 for _ in ('MESSIDOR', 'HRF', 'DRIVE')}
        return trainD, validD, testD


