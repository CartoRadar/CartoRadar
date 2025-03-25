import torch.nn as nn

from .RadarOccNerf import RadarOccNerf

# register the dataset
def get_model(config) -> nn.Module:
    model_name = config.dataset.model_name
    if model_name == 'RadarOccNerf':
        model = RadarOccNerf(config.model)
    else:
        raise ValueError(f'{model_name} incorrect!')
    return model
