import collections.abc
import typing

import numpy as np

from ..utils.typing import ArrayType
from .GeoObject import GeoObject
from .PointSet import PointSet


@GeoObject.register(["polyline", "Polyline"])
class Polyline(PointSet):

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, rank=1, **kwargs)
