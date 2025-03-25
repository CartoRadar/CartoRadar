import attridict, math, torch
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Callable
from . import panoradar_radar

class InfiniteSampler(Sampler):
    def __init__(self, dataset):
        self.dataset_size = len(dataset)

    def __iter__(self):
        while True:
            yield from range(999999)


class ShuffleSampler(Sampler):
    """A custom sampler that shuffles indices and handles epochs internally."""

    def __init__(self, data_source, num_epochs, batch_size):
        self.data_source = data_source
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.total_size = math.ceil(len(self.data_source) / self.batch_size) * self.batch_size

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n).tolist()
        indices += indices[: (self.total_size - len(indices))]
        for i in range(self.num_epochs):
            yield from indices[i * n : (i + 1) * n]

    def __len__(self):
        return self.num_epochs * self.total_size


def get_dataset(config: attridict, model, train=True) -> Tuple[Dataset, Dataset]:
    """Get the corresponding dataset indicated by the config file.
    Returns:
        train_dataset, val_dataset.
    """
    dataset_name = config.dataset.name
    if dataset_name == 'panoradar_radar':
        train_dataset = panoradar_radar.PanoradarRadarDataset(config, model, train=train)
        test_dataset = train_dataset
    else:
        raise ValueError(f'{dataset_name} incorrect!')

    return train_dataset, test_dataset


def get_train_val_functions(config: attridict) -> List[Callable]:
    """Get the functions for training and testing.
    Returns:
        funcs: a list of the following functions:
            - train_step (batch, model, optimizer, lr_scheduler) -> losses
            - train_log (losses, writer, logger, i_step) -> None
            - val_step (model, val_dataset, config) -> results
            - val_log (results, writer, logger, i_step) -> None
    """
    dataset_name = config.dataset.name
    if dataset_name == 'panoradar_radar':
        module = panoradar_radar
    else:
        raise ValueError(f'{dataset_name} incorrect!')

    return (
        module.train_step,
        module.online_train_step,
        module.train_log,
        module.val_step,
        module.val_log,
    )
