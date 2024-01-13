from __future__ import annotations
import collections.abc
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
from ..utils.logger import logger

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


_TNode = typing.TypeVar("_TNode")


class AoS(List[_TNode]):
    """
    Array of structure

    FIXME: 需要优化！！
        - 数据结构应为 named list or ordered dict
        - 可以自动转换 list 类型 cache 和 entry
    """

    def __missing__(self, key) -> _TNode:
        tag = f"@{Path.id_tag_name}"

        if (self._cache is None or len(self._cache) == 0) and self._entry is not None:
            keys = set([key for key in self._entry.child(f"*/{tag}").for_each()])
            self._cache = [{tag: key} for key in keys]
        else:
            self._cache = []

        value = deepcopy(self._metadata.get("default_initial_value", _not_found_) or {})

        value[f"@{Path.id_tag_name}"] = key

        self._cache.append(value)

        return value

    def _update_(self, key, value, *args, **kwargs):
        if key is not None:
            old_value = self._find_(key, default_value=_not_found_)
            new_value = Path._do_change(old_value, value, *args, **kwargs)

            if new_value is not old_value:
                super()._update_(key, new_value, *args, **kwargs)
        elif isinstance(value, list):
            pth = Path(f"@{Path.id_tag_name}")
            for v in value:
                id_tag = pth.get(v, _not_found_)
                if id_tag is _not_found_:
                    self._cache.append(v)
                else:
                    self._update_(id_tag, v, *args, **kwargs)
        else:
            raise TypeError(f"Invalid type of value: {type(value)}")

        return self

    def _for_each_(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        tag = f"@{Path.id_tag_name}"
        for idx, v in enumerate(self._cache):
            key = Path(tag).get(v, _not_found_)
            if key is _not_found_:
                yield self._find_(idx, *args, **kwargs)
            else:
                yield self._find_(key, *args, **kwargs)

            # if self._entry is None:
            #     _entry = None
            # elif key is _not_found_ or v is None:
            #     _entry = self._entry.child(idx)
            # else:
            #     _entry = self._entry.child({tag: key})
            # yield self._type_convert(v, idx, _entry=_entry)

    def fetch(self, *args, _parent=_not_found_, **kwargs) -> Self:
        return self.__duplicate__([HTreeNode._do_fetch(obj, *args, **kwargs) for obj in self], _parent=_parent)

    def dump(self, entry: Entry, **kwargs) -> None:
        """将数据写入 entry"""
        entry.insert([{}] * len(self._cache))
        for idx, value in enumerate(self._cache):
            if isinstance(value, HTree):
                value.dump(entry.child(idx), **kwargs)
            else:
                entry.child(idx).insert(value)
