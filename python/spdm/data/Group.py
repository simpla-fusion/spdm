from .Node import Node
from typing import TypeVar, Mapping
import numpy as np

T = TypeVar('T', str, int, float, np.ndarray, Node)


class Group(Mapping[str, T], Node):
    def __init__(self, data={}, *args,  **kwargs):
        Node.__init__(self, data, *args,   **kwargs)

    def __getitem__(self, key: str) -> T:
        return Node.__getitem__(self, key)

    def __setitem__(self, key: str, value: T) -> None:
        Node.__setitem__(self, key, value)
