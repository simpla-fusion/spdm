from __future__ import annotations

import functools
import typing
from copy import deepcopy
from typing_extensions import Self

from spdm.data.HTree import HTree, HTreeNode
from spdm.utils.tags import _not_found_

from .Entry import Entry
from .HTree import HTree, List, Dict
from .Path import Path, PathLike, as_path, OpTags, update_tree, merge_tree
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import array_type, get_args, get_type_hint

_T = typing.TypeVar("_T")


class QueryResult(HTree):
    """Handle the result of query"""

    def __init__(self, query: PathLike, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._path = as_path(query)

    def __getattr__(self, name: str):
        return self._get(name)

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
    def _value_(self) -> typing.Any:
        value = super()._query(self._path)
        if isinstance(value, list):
            value = functools.reduce(self._default_reducer, value)
        return value

    def __call__(self, *args, **kwargs) -> typing.Any:
        value = super()._query(self._path, op=Path.tags.call, *args, **kwargs)

        if isinstance(value, list):
            value = functools.reduce(self._default_reducer, value)

        return value

    def __iter__(self) -> typing.Generator[typing.Tuple[str, _T | HTree] | _T | HTree, None, None]:
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
            return first + second
        elif isinstance(first, (dict, list)) or isinstance(second, (dict, list)):
            return update_tree(first, second)
        else:
            return first + second

    def children(self) -> typing.Generator[_T | HTree, None, None]:
        """遍历 children"""
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

                yield self._type_convert(value, idx, entry=entry)


_TNode = typing.TypeVar("_TNode", HTreeNode, HTree)


class AoS(List[_TNode]):
    """
    Array of structure

    FIXME: 需要优化！！
        - 数据结构应为 named list or ordered dict
        - 可以自动转换 list 类型 cache 和 entry
    """

    def __iter__(self) -> typing.Generator[_TNode, None, None]:
        for idx, v in self.children():
            yield v

    def __getitem__(self, path) -> _TNode:
        if isinstance(path, str) and path.isidentifier() or isinstance(path, int):
            return self._find(path)
        else:
            return super().__getitem__(path)

    def _find(self, key: PathLike, *args, **kwargs) -> _TNode | QueryResult[_T]:
        """ """
        res = super()._find(key, *args, **kwargs)

        if len(args) == 0 and res is not _not_found_ and isinstance(key, str):
            default_value = merge_tree(
                kwargs.pop("default_value", _not_found_), self._metadata.get("default_initial_value", _not_found_), {}
            )

            self.append({Path.id_tag_name: key, **default_value})

            res = self._find(key, *args, **kwargs)

        return res

        # else:
        #     value = deepcopy(default_value)
        #     pth.update(value, query)
        #     return self._as_child(value, self.__len__())

        # return QueryResult(
        #     query,
        #     self._cache,
        #     default_value=default_value,
        #     _type_hint=self._type_hint(0),
        #     _entry=self._entry,
        #     **kwargs,
        # )

    def fetch(self, *args, _parent=_not_found_, **kwargs) -> Self:
        return self.__duplicate__(
            [HTreeNode._do_fetch(obj, *args, _parent=None, **kwargs) for obj in self], _parent=_parent
        )

    def dump(self, entry: Entry, **kwargs) -> None:
        """将数据写入 entry"""
        entry.insert([{}] * len(self._cache))
        for idx, value in enumerate(self._cache):
            if isinstance(value, HTree):
                value.dump(entry.child(idx), **kwargs)
            else:
                entry.child(idx).insert(value)
