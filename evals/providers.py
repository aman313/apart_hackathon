from inspect_ai.model._registry import modelapi


@modelapi(name='martian_router')
def martian_router():
    from .custom import MartialRouterModel
    return MartialRouterModel

@modelapi(name='martian_base')
def martian_base():
    from .custom import MartianBaseModel
    return MartianBaseModel