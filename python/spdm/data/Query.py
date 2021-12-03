import collections.abc
from copy import deepcopy
from typing import Mapping, TypeVar, Any
from ..util.dict_util import deep_merge_dict
_TQuery = TypeVar("_TQuery", bound="Query")


class Query(object):
    def __init__(self, d: Mapping = None, **kwargs) -> None:
        super().__init__()
        self._query = deep_merge_dict(d, kwargs) if d is not None else kwargs

    def dump(self) -> dict:
        return self._query

    def eval(self, obj) -> Any:
        if len(self._query) == 0:
            return None
        else:
            return NotImplemented

    def update(self, **kwargs):
        self._query.update(kwargs)
