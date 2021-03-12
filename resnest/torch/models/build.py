from fvcore.common.registry import Registry

RESNEST_MODELS_REGISTRY = Registry('RESNEST_MODELS')

def get_model(model_name):
    return RESNEST_MODELS_REGISTRY.get(model_name)
