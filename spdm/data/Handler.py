
from spdm.util.LazyProxy import LazyProxy


class Handler(LazyProxy.Handler):
    DELIMITER = LazyProxy.DELIMITER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MappingHandler(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
