from functools import cached_property, lru_cache

import numpy as np

from ...util.logger import logger

from .Mesh import Mesh


class StructedMesh(Mesh):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args,   **kwargs)

    def axis(self, idx, axis=0):
        return NotImplemented

    def remesh(self, *arg, **kwargs):
        return NotImplemented




