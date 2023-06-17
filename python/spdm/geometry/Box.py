from __future__ import annotations

from .GeoObject import GeoObject
from .BBox import BBox


class Box(GeoObject):
    """ Box
        矩形，一维几何体
    """

    def __init__(self, *args,  **kwargs) -> None:
        bbox = BBox(*args)
        ndim = kwargs.pop("ndim", bbox.ndim)
        super().__init__(ndim=ndim,  **kwargs)
        self._bbox = bbox

    @property
    def bbox(self) -> BBox: return self._bbox
