from .GeoObject import GeoObject, GeoObjectSet
from .Line import Segment
from .Point import Point
import collections.abc


class Polyline(GeoObjectSet[Point]):

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, rank=1, **kwargs)

        self._rank = int(rank) if rank is not None else len(self._points.shape)-1
        self._ndim = ndim if ndim is not None else self._points.shape[-1]

        self._points = np.stack(args) if len(args) != 1 else args[0]

        coordinates = kwargs.get("coordinates", None)
        if coordinates is not None:
            if isinstance(coordinates, str):
                coordinates = [x.strip() for x in coordinates.split(",")]

            if len(coordinates) != self._ndim:
                raise ValueError(f"coordinates {coordinates} not match ndim {self._ndim}")
            elif isinstance(coordinates, collections.abc.Sequence):
                for idx, coord_name in enumerate(coordinates):
                    setattr(self, coord_name, self._points[..., idx])

    def __getitem__(self, *args) -> ArrayType | float: return self._points[args]

    @property
    def points(self) -> typing.List[ArrayType]:
        """ 几何体的点坐标，shape=[npoints,ndim] """
        return tuple([self._points[..., idx] for idx in range(self.ndim)])

    def __array__(self) -> ArrayType: return self._points
    """ 几何体的点坐标，shape=[npoints,ndim] """
