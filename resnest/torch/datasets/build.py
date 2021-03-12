from fvcore.common.registry import Registry

RESNEST_DATASETS_REGISTRY = Registry('RESNEST_DATASETS')

def get_dataset(dataset_name):
    return RESNEST_DATASETS_REGISTRY.get(dataset_name)
