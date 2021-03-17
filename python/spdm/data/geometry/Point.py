from ...util.logger import logger


class Point:
    def __init__(self, *args, **kwargs) -> None:
        self._x = args

    @property
    def topology_rank(self):
        return 1

    @property
    def ndims(self):
        return len(self._x)

    def __call__(self, *args, **kwargs):
        return self._x

    def integrate(self, *args, **kwargs):
        return 0.0

    def apply(self, func, *args, **kwargs):
        return func(*self._x)
