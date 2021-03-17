from ..util.logger import logger


class Point:
    def __init__(self, *args, **kwargs) -> None:
        self._x = args

    @property
    def ndims(self):
        return len(self._x)

    def __call__(self, *args, **kwargs):
        return self._x
