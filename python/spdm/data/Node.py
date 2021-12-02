import collections
import collections.abc

from ..common.logger import logger
from ..common.SpObject import SpObject


class Node(SpObject):
    # __slots__ = "__orig_class__", "_parent"

    def __init__(self, *args, parent=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parent = parent
