from __future__ import annotations

import functools
import typing

from ..utils.tags import _not_found_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import array_type, get_args, get_type_hint
from .Entry import Entry
from .HTree import HTree, List
from .Path import Path, PathLike, as_path

_T = typing.TypeVar("_T")


class QueryResult(HTree[_T]):
    """ Handle the result of query    """

    def __init__(self, query: PathLike, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._path = as_path(query)

    def __getattr__(self, name: str): return self._get(name)

    def _get(self, query: str | int | slice | dict, *args, **kwargs):
        default_value = kwargs.pop("default_value", _not_found_)
        _VT = get_args(self.__orig_class__)[0]
        if isinstance(query, str):

            if default_value is _not_found_ and isinstance(self._default_value, dict):
                default_value = self._default_value.get(query, _not_found_)
            tp = get_type_hint(_VT, query)

            return QueryResult[tp](self._path.append(query), *args, default_value=default_value, **kwargs)
        else:
            return QueryResult[_VT](self._path.append(query), *args, default_value=default_value, **kwargs)

    @property
    def __value__(self) -> typing.Any:
        value = super()._query(self._path)
        if isinstance(value, list):
            value = functools.reduce(self._default_reducer, value)
        return value

    def __call__(self, *args, **kwargs) -> typing.Any:

        value = super()._query(self._path, op=Path.tags.call, *args, **kwargs)

        if isinstance(value, list):
            value = functools.reduce(self._default_reducer, value)

        return value

    def __iter__(self) -> typing.Generator[typing.Tuple[str, _T | HTree[_T]] | _T | HTree[_T], None, None]:
        raise NotImplementedError(f"TODO:")

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

    def children(self) -> typing.Generator[_T | HTree[_T], None, None]:
        """ 遍历 children """
        cache = self._cache if self._cache is not _not_found_ else self._default_value

        if not isinstance(cache, list) or len(cache) == 0:
            yield from super().children()

        else:
            for idx, value in enumerate(cache):
                if isinstance(value, (dict, Dict)):
                    id = value.get(self._identifier, None)
                else:
                    id = None
                if id is not None:
                    entry = self._entry.child({f"@{self._identifier}": id})
                else:
                    entry = None

                yield self._as_child(value, idx, entry=entry)


class AoS(List[_T]):
    """
        Array of structure
    """

    def __init__(self, *args, identifier: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._identifier = identifier
        if self._identifier is None:
            self._identifier = self.__metadata__.get("identifier", None)

    def dump(self, entry: Entry, **kwargs) -> None:
        """ 将数据写入 entry """
        entry.insert([{}]*len(self._cache))
        for idx, value in enumerate(self._cache):
            if isinstance(value, HTree):
                value.dump(entry.child(idx), **kwargs)
            else:
                entry.child(idx).insert(value)

    def _get(self, query: PathLike,  **kwargs) -> HTree[_T] | _T | QueryResult[_T]:

        if isinstance(query, int):
            return super()._get(query)

        elif isinstance(query, str):
            query = {f"@{self._identifier}": query}

        elif not isinstance(query, (slice, dict)):
            raise TypeError(f"{type(query)}")

        default_value = kwargs.pop("default_value", self._default_value)

        tp = self._type_hint(0)

        return QueryResult[tp](query, self._cache, entry=self._entry, default_value=default_value, parent=self._parent, **kwargs)

# class NamedDict(HTree[_T]):
#     """ Proxy to access named dict """

#     def __getattr__(self, name: str) -> typing.Any: return self._get(name)

# class QueryResult(HTree):
#     """ Handle the result of query    """

#     def __init__(self, cache: list,  *args,  **kwargs) -> None:
#         super().__init__(cache,  *args, **kwargs)

#     def __getattr__(self, name: str) -> QueryResult:
#         if isinstance(self._default_value, dict):
#             default_value = self._default_value.get(name, _not_found_)
#         else:
#             default_value = _not_found_
#         return QueryResult([v.get(name, _not_found_) for v in self._cache if v is not _not_found_], default_value=default_value)

#     def __getitem__(self, query: PathLike) -> QueryResult:
#         return QueryResult([v.get(query, _not_found_) for v in self._cache if v is not _not_found_], default_value=self._default_value)

#     def __iter__(self) -> typing.Generator[typing.Tuple[str, typing.Any | HTree] | typing.Any | HTree, None, None]:
#         if not isinstance(self._default_value, list):
#             raise NotImplementedError(f"default_value={self._default_value}")
#         for v in self._default_value:
#             logger.debug(v)
#             yield v
#         # return super().__iter__()

#     ###########################################################

#     @property
#     def __value__(self) -> typing.List[typing.Any]:
#         value = [as_value(v) for v in self._cache]
#         if all([v is _not_found_ for v in value]) or len(value) == 0:
#             return _not_found_
#         else:
#             return value

#     def __reduce__(self) -> typing.Any:

#         value = self.__value__

#         if not isinstance(value, list):
#             value = [value]

#         return functools.reduce(self._default_reducer,  value)

#     @staticmethod
#     def _default_reducer(first: typing.Any, second: typing.Any) -> typing.Any:

#         if first is _not_found_:
#             return second
#         elif second is _not_found_ or second is None:
#             return second
#         elif isinstance(first, (str)):
#             return first
#         elif isinstance(first, array_type) and isinstance(second, array_type):
#             return first+second
#         elif isinstance(first, (dict, list)) or isinstance(second, (dict, list)):
#             return merge_tree_recursive(first, second)
#         else:
#             return first+second

#     def _op_call(self, *args, **kwargs) -> typing.Any:
#         value = [(v(*args, **kwargs) if callable(v) else v) for v in self._cache if v is not _not_found_]
#         if len(value) == 0:
#             raise RuntimeError(f"TODO: suffix={self._suffix} not found!")
#         return functools.reduce(self._default_reducer, value)
