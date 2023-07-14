from __future__ import annotations

import collections.abc
import inspect
import typing
from copy import copy, deepcopy
from functools import reduce

from ..utils.logger import logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (ArrayType, HNodeLike, NumericType, PrimaryType,
                            array_type, as_array, as_value, get_args,
                            get_origin, isinstance_generic, numeric_type,
                            primary_type, serialize, type_convert)
from .Entry import Entry, as_entry
from .Expression import Expression
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

    def __getattr__(self, name: str) -> typing.Any: return self._lazy_get(name)

    def __getitem__(self, path: PathLike) -> typing.Any: return self._lazy_get(path)

    def __setitem__(self, path: PathLike, value): raise NotImplementedError(f"TODO: setitem {path} {value}")

    @ property
    def current(self) -> typing.Any: return self._lazy_get(-1)

    ###################################################################################

    def _type_hint(self, path=None) -> typing.Type:
        suffix = copy(self._suffix)
        suffix.append(path)
        return self._target._type_hint(suffix)

    def _lazy_get(self, path) -> QueryResult | PrimaryType:
        other = copy(self)
        other._suffix.append(path)
        type_hint = self._target._type_hint([0]+other._suffix[:])
        if not issubclass(get_origin(type_hint), HTree):
            return type_hint(other.__reduce__())
        else:
            return other

    # def __setattr__(self, path: PathLike, value):
    #     raise NotImplementedError(f"TODO: setitem {path} {value}")

    def __contain__(self, value):
        raise NotImplementedError(f"TODO: __contain__ {value}")

    def __len__(self, value):
        raise NotImplementedError(f"TODO: __len__ {value}")

    def __iter__(self) -> typing.Generator[QueryResult, None, None]:

        default_value = self._suffix.collapse().query(self._target._default_value)

        if not isinstance(default_value, collections.abc.Sequence) or not isinstance(default_value[0], collections.abc.Mapping):
            raise NotImplementedError(f"TODO: __iter__ {default_value} {self._suffix}")

        identifier = "label"

        for v in default_value:

            id = v.get(identifier, None)

            suffix = copy(self._suffix)+[{f"@{identifier}": id}]

            yield QueryResult(self._target, self._query_cmd, suffix=suffix, reducer=self._reducer)

    def for_each(self,  start: PathLike = None, *args, **kwargs) -> typing.Tuple[Entry, PathLike]:

        if suffix is None:
            suffix = self._suffix
        else:
            suffix = as_path(suffix)

        next_id = None

        while True:
            obj, next_id = self._target._find_next(self._query_cmd, start=next_id)

            if next_id is None:
                break

            elif len(suffix) == 0:
                value = obj

            elif isinstance(obj, HTree):
                value = obj.get(suffix)

            else:
                value = suffix.query(obj, default_value=_not_found_)

            yield value

    ###########################################################
    # API: CRUD  operation

    def query(self, op=None, *args, **kwargs) -> typing.Any:
        raise NotImplementedError(f"TODO: query {op} {args} {kwargs}")

    def insert(self, *args, **kwargs) -> Entry:
        raise NotImplementedError(f"TODO: insert {args} {kwargs}")

    def update(self, *args, **kwargs) -> Entry:
        raise NotImplementedError(f"TODO: update {args} {kwargs}")

    def find_next(self,  start: PathLike = None, *args, **kwargs) -> typing.Tuple[Entry, PathLike]:
        raise NotImplementedError(f"TODO: find_next {args} {kwargs}")

    def remove(self, *args, **kwargs) -> int:
        raise NotImplementedError(f"TODO: insert {args} {kwargs}")
    ###########################################################

    @ property
    def __value__(self) -> typing.List[typing.Any]:
        value = [as_value(v) for v in self._foreach() if v is not _not_found_]
        if len(value) == 0 or isinstance(value[0], collections.abc.Mapping)\
                and self._target._default_value is not _not_found_ and len(self._suffix) > 0:
            default_value = self._suffix.collapse().query(self._target._default_value)
            value = [default_value]+value
        return value

    def __reduce__(self) -> typing.Any:
        
        return reduce(self._reducer,  self.__value__)

    @ staticmethod
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
