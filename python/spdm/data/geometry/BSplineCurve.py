
from scipy.interpolate import splev, splprep, splrep

from ...util.logger import logger
from ..geometry.Curve import Curve


class BSplineCurve(Curve):
    def __init__(self,  *args, is_closed=None, **kwargs) -> None:
        if len(args) != 2:
            raise ValueError(f"Illegal input! len(args)={len(args)}")
        super().__init__(is_closed=is_closed is not None, **kwargs)

        if self.is_closed:
            self._spl, _ = splprep(args, s=0)
        else:
            self._spl = splrep(*args, s=0)

    def inside(self, *x):
        return NotImplemented

    def points(self, u, *args, **kwargs):
        if self.is_closed:
            return splev(u, self._spl, *args, **kwargs)
        else:
            return splev(u, self._spl, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.points(*args, **kwargs)
