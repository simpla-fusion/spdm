from itertools import cycle

from scipy.interpolate import splev, splprep, splrep

from ...util.logger import logger
from ..Curve import Curve


class BSplineCurve(Curve):
    def __init__(self, X, Y, *args, cycle=None, **kwargs) -> None:
        super().__init__(*args, is_closed=cycle is not None, **kwargs)
        if self.is_closed:
            self._spl, _ = splprep([X, Y], s=0)
            logger.debug(_)
        else:
            self._spl = splrep(X, Y, s=0)

    def inside(self, *x):
        return NotImplemented

    def xy(self, u, *args, **kwargs):
        if self.is_closed:
            return splev(u, self._spl, *args, **kwargs)
        else:
            return splev(u, self._spl, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.xy(*args, **kwargs)
