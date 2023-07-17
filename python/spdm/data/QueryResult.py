from __future__ import annotations

import collections.abc
import typing
from copy import copy
from functools import reduce

from ..utils.tags import _not_found_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (PrimaryType,
                            array_type, as_value, get_origin)
from .Entry import Entry
from .Path import Path, PathLike, as_path

_T = typing.TypeVar("_T")


class QueryResult(Entry):
    """ Handle the result of query    """

    def __init__(self, target: typing.Any, query: PathLike,  *args,
                 reducer: typing.Callable[..., typing.Any] | None = None, **kwargs) -> None:
        super().__init__(target, query, *args, **kwargs)
        self._reducer = reducer if reducer is not None else QueryResult._default_reducer

    def __copy_from__(self, other: QueryResult) -> QueryResult:
        super().__copy_from__(other)
        self._reducer = other._reducer
        return self

    ###################################################################################

    def __equal__(self, other) -> bool:
        if isinstance(other, Entry):
            return other._data == self._data and other._path == self._path
        else:
            return self.query(Path.tags.equal, other)

    @property
    def count(self) -> int: raise NotImplementedError(f"TODO: count {self._path}")

    @property
    def exists(self) -> bool:
        return any([e.query(Path.tags.exists) for e in self._foreach() if isinstance(e, Entry)])

    def check_type(self, tp: typing.Type) -> bool:
        return any([not e.query(Path.tags.check_type, tp) for e in self._foreach() if isinstance(e, Entry)])

    def dump(self) -> typing.Any:
        return self.__reduce__([e.query(Path.tags.dump) for e in self._foreach() if isinstance(e, Entry)])

    def get(self, *args, default_value: typing.Any = ..., **kwargs) -> typing.Any:
        res = [e.get(*args, default_value=_not_found_, **kwargs) for e in self._foreach() if isinstance(e, Entry)]

        res = [e for e in res if e is not _not_found_]

        if len(res) == 0:
            res = [default_value]

        return self.__reduce__(res)

    ###########################################################
    # API: CRUD  operation

    def query(self, op=None, *args, **kwargs) -> typing.Any:
        return [v.query(op, *args, **kwargs) for v in self._foreach() if v is not _not_found_]

    def insert(self, *args, **kwargs) -> Entry:
        raise NotImplementedError(f"TODO: insert {args} {kwargs}")

    def update(self, *args, **kwargs) -> Entry:
        raise NotImplementedError(f"TODO: update {args} {kwargs}")

    def remove(self, *args, **kwargs) -> int:
        raise NotImplementedError(f"TODO: insert {args} {kwargs}")

    ###########################################################

    def _foreach(self, *args, **kwargs) -> typing.Generator[Entry, None, None]:
        next_id = []
        while True:
            entry, next_id = self.find_next(*next_id,   **kwargs)
            if next_id is None:
                break
            yield entry

    @property
    def __value__(self) -> typing.List[typing.Any]:
        value = [as_value(v) for v in self._foreach() if v is not _not_found_]
        if len(value) == 0:
            return _not_found_
        else:
            return value

    def __reduce__(self, value=None) -> typing.Any:
        if value is None:
            value = self.__value__
        if not isinstance(value, list):
            value = [value]
        return reduce(self._reducer,  value)

    @staticmethod
    def _default_reducer(first: typing.Any, second: typing.Any) -> typing.Any:

        if first is _not_found_:
            return second
        elif second is _not_found_ or second is None:
            return second
        elif isinstance(first, (str)):
            return first
        elif isinstance(first, array_type) and isinstance(second, array_type):
            return first+second
        elif isinstance(first, (dict, list)) or isinstance(second, (dict, list)):
            return merge_tree_recursive(first, second)
        else:
            return first+second

    def _op_call(self, *args, **kwargs) -> typing.Any:
        value = [(v(*args, **kwargs) if callable(v) else v) for v in self._foreach() if v is not _not_found_]
        if len(value) == 0:
            raise RuntimeError(f"TODO: suffix={self._suffix} not found!")
        return reduce(self._reducer, value)
