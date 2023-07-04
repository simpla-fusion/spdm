from __future__ import annotations

import collections.abc
import typing
from copy import copy, deepcopy
from functools import reduce


from ..utils.logger import logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (ArrayType, PrimaryType, array_type, as_array,
                            get_args, get_origin, isinstance_generic,
                            numeric_type, serialize, type_convert, HTreeLike, HNodeLike)
from .Entry import Entry, as_entry
from .Path import Path, PathLike, as_path


_T = typing.TypeVar("_T")


class HTree(typing.Generic[_T]):
    """
        Hierarchical Tree:

        一种层次化的数据结构，它具有以下特性：
        - 树节点也可以是列表 list，也可以是字典 dict
        - 叶节点可以是标量或数组 array_type，或其他 type_hint 类型
        - 节点可以有缓存（cache)
        - 节点可以有父节点（parent)
        - 节点可以有元数据（metadata)
            - 包含： 唯一标识（id), 名称（name), 单位（units), 描述（description), 标签（tags), 注释（comment)
        - 任意节点都可以通过路径访问
        - 泛型 _T 变量，为 element 的类型

        @NOTE:
            - Node,Dict,List 不缓存__getitem__结果
            - __getitem__ 返回的类型由 __type_hint__ 决定，默认为 Node
        -
    """

    def __init__(self, data: HTreeLike | Entry = None,  cache: typing.Any = None,
                 parent: HTree | None = None, key: PathLike = None, **kwargs) -> None:

        self._entry = as_entry(data)
        self._cache = cache
        self._parent = parent
        self._key = key
        self._metadata = kwargs.pop("metadata", {}) | kwargs

    def __copy__(self) -> HTree[_T]:
        other: HTree = self.__class__.__new__(getattr(self, "__orig_class__", self.__class__))
        other._entry = copy(self._entry)
        other._parent = self._parent
        other._key = None
        other._metadata = copy(self._metadata)
        other._cache = copy(other._cache)
        return other

    def __serialize__(self) -> typing.Any: return serialize(self.__value__)

    @classmethod
    def __deserialize__(cls, *args, **kwargs) -> HTree: return cls(*args, **kwargs)

    def __str__(self) -> str: return f"<{self.__class__.__name__} />"

    @property
    def __value__(self) -> typing.Any: return self._get_by_query()

    def __array__(self) -> ArrayType: return as_array(self._get_by_query())

    def _repr_svg_(self) -> str:
        from ..views.View import display
        return display(self, output="svg")

    @property
    def __name__(self) -> str: return self._metadata.get("name", "unamed")

    @property
    def __metadata__(self) -> dict: return self._metadata

    def __getitem__(self, path) -> HTree[_T] | _T | PrimaryType: return self.get(path, default_value=_undefined_)

    def __setitem__(self, path, value) -> None: self._update(path, value)

    def __delitem__(self, path) -> int: return self._remove(path)

    def __contains__(self, key) -> bool: return self._query([key], Path.tags.exists)  # type:ignore

    def __len__(self) -> int: return self._query([], Path.tags.count)  # type:ignore

    def __iter__(self) -> typing.Generator[HTree[_T], None, None]:
        """ 遍历 children """
        type_hint = self._type_hint()
        parent = self._parent
        default_value = self.__metadata__.get("default_vlaue", _not_found_)
        key: int | None = None

        while True:

            value, key = self._find_next(start=key, type_hint=type_hint, parent=parent,
                                         default_value=default_value)

            if key is None:
                break

            yield value

    def __next__(self) -> HTree[_T]:
        """ 遍历 slibings """
        if self._parent is None:
            raise StopIteration(f"{self} has no parent!")

        value, key = self._parent._find_next(start=self._key)

        if key is None:
            raise StopIteration(f"{self} has no next!")

        return value

    def __equal__(self, other) -> bool: return self._query([], Path.tags.equal, other)  # type:ignore

    def insert(self, *args, **kwargs): return self._insert([], *args, **kwargs)

    def update(self, *args, **kwargs): return self._update([], *args, **kwargs)

    def get(self, path: Path | PathLike, *args,
            type_hint: typing.Type = None,
            default_value: typing.Any = _not_found_,
            **kwargs) -> HTree[_T] | _T | PrimaryType:

        path = as_path(path)

        if len(path) == 0:
            return self

        obj = self
        pos = 0
        for idx, p in enumerate(path):
            pos = idx
            if isinstance(obj, HTree):
                if idx == len(path)-1 and type_hint is not None:
                    obj = obj._get_by_query(p, type_hint=type_hint, default_value=default_value)
                else:
                    obj = obj._get_by_query(p, default_value=_not_found_)
            else:
                obj = Path(path[idx:]).query(obj, default_value=_not_found_)
                obj = type_convert(obj, type_hint=type_hint, default_value=default_value, **kwargs)
                pos = len(path)-1
                break
        else:
            if obj is _not_found_:
                raise KeyError(f"{path[:pos+1]} not found")

        if obj is _not_found_:
            obj = default_value

        if obj is _undefined_:
            raise KeyError(f"Can't find {path}!")

        return obj

    ################################################################################
    # Private methods

    def _type_hint(self, key: PathLike = None) -> typing.Type | None:
        """ 当 key 为 None 时，获取泛型参数，若非泛型类型，返回 None，
            当 key 为字符串时，获得属性 property 的 type_hint
        """
        tp = get_origin(self)

        tp_hint = None
        if isinstance(key, str):
            tp_hint = typing.get_type_hints(tp).get(key, None)

        if tp_hint is None:
            tp_hint = get_args(self)
            tp_hint = None if len(tp_hint) == 0 else tp_hint[-1]
        return tp_hint

    def _update_cache(self, key: PathLike, value: typing.Any = _not_found_,
                      type_hint=None,
                      default_value: typing.Any = _not_found_,
                      parent=None,  **kwargs) -> HTree[_T]:

        if self._cache is None and value is _not_found_:
            return _not_found_

        elif self._cache is None:
            self._cache: typing.Dict[str | int, typing.Any] = {}

        if type_hint is None:
            type_hint = self._type_hint(key)

        if value is _not_found_:
            value = self._cache.get(key, _not_found_)

        if value is _not_found_:
            pass
        elif not isinstance_generic(value, type_hint):
            value = type_convert(value, type_hint=type_hint,
                                 default_value=default_value,
                                 parent=parent, key=key, **kwargs)
            self._cache[key] = value

        return value

    def _get_by_query(self, query: PathLike = None, type_hint=None, **kwargs) -> HTree[_T]:
        """ 获取子节点  """

        value = _not_found_

        if query is None and (self._cache is None or self._cache is _not_found_):  # get value from self._entry and update cache

            value = self._entry.get(default_value=_not_found_)

            default_value = deepcopy(self._metadata.get("default_value", _not_found_))

            if isinstance(default_value, dict) and len(default_value) > 0:
                Path().update(default_value, value)
                value = default_value

            if value is not _not_found_:
                self._cache = value

        elif query is None:  # return self._cache
            value = self._cache

        elif query is Path.tags.current:
            value = self

        elif query is Path.tags.parent:
            value = self._parent

        elif query is Path.tags.next:
            raise NotImplementedError(f"TODO: operator 'next'!")
            # value = self._parent.next(self)

        elif query is Path.tags.root:
            value = self._root

        elif isinstance(query, (int, slice, tuple)) and type_hint in numeric_type:
            if self._cache is None or len(self._cache) == 0:
                self._cache = copy(self._entry.__value__)  # type:ignore

            if isinstance(self._cache, array_type) or isinstance(self._cache, collections.abc.Sequence):
                value = self._cache[query]

            elif self._cache is None or self._cache is _not_found_:
                return kwargs.get("default_value", _not_found_)  # type:ignore
            else:
                raise RuntimeError(f"{self._cache}")

        elif isinstance(query, str):
            value = self._get_by_name(query, type_hint=type_hint, **kwargs)

        elif isinstance(query, int):
            value = self._get_by_index(query, type_hint=type_hint, **kwargs)

        elif isinstance(query, set):
            value = NamedDictProxy(cache={k: self._get_by_name(k, type_hint=type_hint, **kwargs) for k in query})

        else:  # as query return QueryResult
            value = QueryResult(self, query, type_hint=type_hint, parent=self._parent)

        return value  # type:ignore

    def _get_by_name(self, key: str,  default_value=_not_found_, type_hint=None, getter=None, **kwargs) -> HTree[_T]:

        if type_hint is None:
            type_hint = self._type_hint(key) or HTree[_T]

        if default_value is _not_found_:
            default_value = self._metadata.get("default_value", _not_found_)
            if isinstance(default_value, dict):
                default_value = default_value.get(key, _not_found_)

        value = self._update_cache(key,  type_hint=type_hint,
                                   default_value=default_value,
                                   parent=self,
                                   **kwargs)

        if value is not _not_found_:
            return value

        if getter is not None:
            value = getter(self)

        if value is _not_found_ and isinstance(self._entry, Entry):
            value = self._entry.child(key)

        if value is not _not_found_ and not isinstance(value, HTree):
            value = self._update_cache(key,  value,
                                       type_hint=type_hint,
                                       default_value=default_value,
                                       parent=self,  **kwargs)

        return value

    def _get_by_index(self, key: int,  default_value=_not_found_, type_hint=None, **kwargs) -> HTree[_T]:

        if type_hint is None:
            type_hint = self._type_hint() or HTree[_T]

        if default_value is _not_found_:
            default_value = self._metadata.get("default_value", {})

        if key < 0 and self._entry is not None:
            key += self._entry.count

        value = self._update_cache(key,  type_hint=type_hint,
                                   default_value=default_value,
                                   parent=self._parent, **kwargs)

        if value is _not_found_ and isinstance(self._entry, Entry):
            value = self._update_cache(key, self._entry.child(key),
                                       type_hint=type_hint,
                                       default_value=default_value,
                                       parent=self._parent, **kwargs)
        return value

    def _query(self,  path: PathLike,   *args,  **kwargs) -> HTree[_T] | PrimaryType:
        return self._entry.child(path).query(*args, **kwargs)
        path = as_path(path)

        default_value = kwargs.pop("default_value", _not_found_)

        missing_cache = True

        value = _not_found_

        if self._cache is not None:
            value = path.query(self._cache, *args, default_value=_not_found_, **kwargs)

        if value is not _not_found_:
            missing_cache = False
        elif self._entry is not None:
            value = path.query(self._entry, *args, default_value=_not_found_,  **kwargs)

        # if default_value is _not_found_:
        #     default_value = path.query(self._metadata.get("default_value", _not_found_), default_value=_not_found_)

        # parent = self if isinstance(path[0], str) else self._parent

        # n_value = type_convert(value, self.__type_hint__(path), default_value=default_value, parent=parent)

        # if missing_cache:
        #     path.insert(self, value)

        return value

    def _insert(self, path: PathLike,   *args, **kwargs) -> HTree[_T]:
        # try:
        #     new_path = Path().insert(self._cache, *args, quiet=False, **kwargs)
        #     res = self[new_path]
        # except KeyError:
        self._entry.child(path).insert(*args, quiet=True,  **kwargs)
        return self

    def _update(self, path: PathLike,  *args, **kwargs) -> HTree[_T]:
        # try:
        #     Path().update(self._cache, *args, quiet=False, **kwargs)
        # except KeyError:
        self._entry.child(path).update(*args, quiet=True,  **kwargs)
        return self

    def _remove(self, path: PathLike,  *args, **kwargs) -> int:
        return self._entry.child(path).remove(*args, **kwargs)

    @property
    def _root(self) -> HTree | None:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while p._parent is not None and getattr(p, "ids_properties", None) is None:
            p = p._parent
        return p

    def _find_next(self, query: PathLike = None, start: PathLike = None, *args, **kwargs) -> typing.Tuple[HTree[_T], PathLike]:
        if query is None:
            return self._find_by_slice(slice(None), start=start, *args, **kwargs)
        elif isinstance(query, slice):
            return self._find_by_slice(query, start=start, *args, **kwargs)
        elif isinstance(query, dict):
            return self._find_by_query(query, start=start, *args, **kwargs)
        else:
            raise NotImplementedError(f"TODO: _find {query}!")

    def _find_by_query(self, query: PathLike = None, start: PathLike = None, *args, **kwargs) -> typing.Tuple[HTree[_T], int]:
        raise NotImplementedError(f"TODO: _find_by_query {query}!")
        type_convert(value, type_hint=type_hint, parent=parent)

    def _find_by_slice(self, s: slice, start: PathLike = None, *args,  **kwargs) -> typing.Tuple[HTree[_T], int | None]:

        if start is None:
            start = s.start or 0
        elif s.start is not None and start < s.start:
            raise IndexError(f"Out of range: {start} < {s.start}!")

        stop = s.stop

        step = s.step or 1

        if stop is not None and start >= stop:
            return None, None

        value = _not_found_

        if isinstance(self._cache, dict):
            value = self._cache.get(start, _not_found_)

        if value is _not_found_:
            value = self._entry.child(start)
            if not value.exists:
                start = None
                value = None

        if value is not _not_found_ and start is not None:
            value = self._update_cache(start, value, *args,   **kwargs)
            start += step

        return value, start


def as_htree(obj, *args, **kwargs):
    if isinstance(obj, HTree):
        return obj
    else:
        return HTree(obj, *args, **kwargs)


Node = HTree


class Container(HTree[_T]):
    def __getitem__(self, path) -> HTree[_T] | _T | PrimaryType: return self.get(path)


class Dict(Container[_T]):
    def __getitem__(self, path) -> HTree[_T] | _T | PrimaryType: return self.get(path)


class List(Container[_T]):
    def __getitem__(self, path) -> HTree[_T] | _T | PrimaryType: return self.get(path)


class NamedDictProxy(HTree[_T]):
    def __getattr__(self, name: str) -> typing.Any: return self._get_by_query(name)


class QueryResult:
    """ Handle the result of query    """

    def __init__(self, target: HTree, query: PathLike,  *args, suffix: PathLike | Path = None,
                 reducer: typing.Callable[..., HTreeLike] | None = None,
                 **kwargs) -> None:
        # super().__init__(*args, **kwargs)
        self._query_cmd = query
        self._suffix = as_path(suffix)
        self._target = as_htree(target)
        self._reducer = reducer if reducer is not None else QueryResult._default_reducer

    def __copy__(self) -> QueryResult:
        other = QueryResult(self._target, self._query_cmd, suffix=self._suffix, reducer=self._reducer)
        return other

    def __getattr__(self, name: str) -> typing.Any: return self._lazy_get(name)

    def __getitem__(self, path: PathLike) -> typing.Any: return self._lazy_get(path)

    def _lazy_get(self, path) -> QueryResult:
        new_path = copy(self._suffix)
        new_path.append(path)
        return QueryResult(self._target, new_path)

    def __setitem__(self, path: PathLike, value):
        raise NotImplementedError(f"TODO: setitem {path} {value}")

    # def __setattr__(self, path: PathLike, value):
    #     raise NotImplementedError(f"TODO: setitem {path} {value}")

    def __contain__(self, value):
        raise NotImplementedError(f"TODO: __contain__ {value}")

    def __len__(self, value):
        raise NotImplementedError(f"TODO: __len__ {value}")

    def __iter__(self, value) -> typing.Generator[HTreeLike, None, None]:
        raise NotImplementedError(f"TODO: __iter__ {value}")

    def _foreach(self) -> typing.Generator[HTreeLike | HTree, None, None]:
        type_hint = self._target._type_hint()
        default_value = self._target.__metadata__.get("default_value", _not_found_)
        parent = self._target._parent
        start = None

        while True:
            value, start = self._target._find_next(self._query_cmd, start=start,
                                                   type_hint=type_hint, parent=parent,
                                                   default_value=default_value)
            if start is None:
                break

            elif len(self._suffix) == 0:
                pass

            elif isinstance(value, HTree):
                value = value[self._suffix]

            else:
                value = self._suffix.query(value, default_value=_not_found_)

            yield value

    @property
    def __value__(self) -> typing.List[HTreeLike]: return [as_value(v) for v in self._foreach() if v is not _not_found_]

    @property
    def __reduce__(self) -> HTreeLike: return reduce(self._reducer, self.__value__)

    @staticmethod
    def _default_reducer(first: HTreeLike, second: HTreeLike) -> HTreeLike:

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


class AoS(List[_T]):
    """
        Array of structure
    """

    def __init__(self, *args, identifier: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._identifier = identifier if identifier is not None else self.__metadata__.get("identifier", "id")

    def _get_by_query(self, query: PathLike = None,  *args, **kwargs) -> HTree[_T]:
        if isinstance(query, (str, set)):
            query = {f"@{self._identifier}": query}

        return super()._get_by_query(query,  *args,  **kwargs)