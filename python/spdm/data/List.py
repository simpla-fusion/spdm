import collections

from matplotlib.pyplot import loglog
from spdm.util.logger import logger
from spdm.data.Node import Node

from .Group import Group


class List(Group):
    def __init__(self, d=None, *args, default_factory=None, parent=None, **kwargs):

        super().__init__(None, *args,  parent=parent, **kwargs)
        self._default_factory = default_factory
        if d is not None:
            self._data = [self.__new_child__(v) for v in d]

    def __new_child__(self, *args, parent=None, **kwargs):
        if self._default_factory is None:
            return super().__new_child__(*args, parent=parent or self._parent, **kwargs)
        else:
            return self._default_factory(*args, parent=parent or self._parent, **kwargs)

    def __iter__(self):
        yield from self._data
