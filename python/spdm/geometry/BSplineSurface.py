

from ..data.Function import Function
from ..common.logger import logger
from .Surface import Surface


class BSplineSurface(Surface):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise NotImplementedError()
