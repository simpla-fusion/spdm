from __future__ import annotations

import collections.abc
import functools
import inspect
import typing
from copy import copy, deepcopy

from ..utils.logger import deprecated, logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (ArrayType, HTreeLike, NumericType, array_type,
                            as_array, as_value, get_args, get_origin,
                            get_type_hint, isinstance_generic, numeric_type,
                            serialize, type_convert)
from .Entry import Entry, as_entry
from .Path import Path, PathLike, Query, QueryLike, as_path, as_query

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

    def __init__(self, cache: typing.Any = None, /,  entry: Entry | None = None, parent: HTree | None = None, **kwargs) -> None:

        default_value = _not_found_

        if isinstance(cache, dict):
            default_value = merge_tree_recursive(default_value, cache.pop("$default_value", _not_found_))

        default_value = merge_tree_recursive(default_value, kwargs.pop("default_value", _not_found_))

        if cache is None or cache is _undefined_:
            cache = _not_found_
        self._cache = cache
        self._entry = as_entry(entry)
        self._default_value = deepcopy(default_value)
        self._metadata = kwargs.pop("metadata", {}) | kwargs
        self._parent = parent
        self._id = self._metadata.get("id", None) or self._metadata.get("name", None)

    def __copy__(self) -> HTree[_T]:
        other: HTree = self.__class__.__new__(getattr(self, "__orig_class__", self.__class__))
        other.__copy_from__(self)
        return other

    def __copy_from__(self, other: HTree[_T]) -> HTree[_T]:
        """ 复制 other 到 self  """
        if isinstance(other, HTree):
            self._cache = copy(other._cache)
            self._entry = copy(other._entry)
            self._parent = other._parent
            self._metadata = copy(other._metadata)
            self._default_value = copy(other._default_value)
        return self

    def __serialize__(self) -> typing.Any: return serialize(self.__value__)

    @classmethod
    def __deserialize__(cls, *args, **kwargs) -> HTree: return cls(*args, **kwargs)

    def __str__(self) -> str: return f"<{self.__class__.__name__} />"

    @property
    def __value__(self) -> typing.Any:
        if self._cache is _not_found_:
            self._cache = merge_tree_recursive(self._default_value, self._entry.get(default_value=_not_found_))
        return self._cache

    def __array__(self) -> ArrayType: return as_array(self.__value__)

    def _repr_svg_(self) -> str:
        from ..view.View import display
        return display(self, output="svg")

    # def __reduce__(self) -> _T: raise NotImplementedError(f"")

    @property
    def __name__(self) -> str: return self._metadata.get("name", "unamed")

    @property
    def __metadata__(self) -> dict: return self._metadata

    @property
    def _root(self) -> HTree[_T] | None:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while p._parent is not None and getattr(p, "ids_properties", None) is None:
            p = p._parent
        return p

    def __getitem__(self, path) -> _T: return self.get(path, force=True)

    def __setitem__(self, path, value) -> None: self._update(path, value)

    def __delitem__(self, path) -> None: return self._remove(path)

    def __contains__(self, key) -> bool: return self._query([key], Path.tags.exists)  # type:ignore

    def __len__(self) -> int: return self._query([], Path.tags.count)  # type:ignore

    def __iter__(self) -> typing.Generator[_T | HTree[_T], None, None]: yield from self.children()
    """ 遍历 children """

    def __equal__(self, other) -> bool: return self._query([], Path.tags.equal, other)  # type:ignore

    # def children(self) -> typing.Generator[typing.Any, None, None]: yield from self._foreach()
    # """ 遍历 children """

    def insert(self,  *args, **kwargs): return self._insert([], *args, **kwargs)

    def update(self, *args, **kwargs): return self._update([], *args, **kwargs)

    def remove(self, *args, **kwargs): return self._remove(*args, **kwargs)

    @deprecated
    def refresh(self, cache=None, **kwargs):
        if cache is None:
            cache = kwargs
        if isinstance(cache, dict):
            cache = merge_tree_recursive(self._default_value, cache)
        # logger.debug(self._cache)
        # logger.debug(self._default_value)
        self._cache = cache
        return self

    def get(self, path: Path | PathLike,  default_value: typing.Any = _not_found_, *args,   force=False, **kwargs) -> _T:

        path = as_path(path)
        length = len(path)

        if length == 0:
            if force:
                return self
            else:
                return self.__value__

        obj = self

        pos = -1

        for idx, p in enumerate(path[:-1]):

            if isinstance(obj, HTree):
                tmp = obj._get(p, default_value=_not_found_, force=True)
                pos = idx
            else:
                tmp = Path(path[idx:]).fetch(obj, default_value=_not_found_)
                pos = len(path)
                break

            if tmp is _not_found_ or pos >= length:
                break
            else:
                obj = tmp

        if isinstance(obj, HTree) and pos == length-2:
            obj = obj._get(path[-1], *args, default_value=default_value, force=force, **kwargs)

        if obj is _not_found_:
            obj = default_value

        if obj is _undefined_ and pos <= len(path):
            raise KeyError(f"{path[:pos+1]} not found")

        return obj

    def children(self) -> typing.Generator[_T, None, None]:

        if isinstance(self._cache, list) and len(self._cache) > 0:
            for idx, cache in enumerate(self._cache):
                yield self._as_child(cache, idx, entry=self._entry.child(idx))
        elif isinstance(self._cache, dict) and len(self._cache) > 0:
            for key, cache in self._cache.items():
                yield self._as_child(cache, key, entry=self._entry.child(key))
        else:
            for key, entry in self._entry.for_each():
                yield self._as_child(None, key, entry=entry)

    ################################################################################
    # Private methods

    def _type_hint(self, path: PathLike = None) -> typing.Type:
        """ 当 key 为 None 时，获取泛型参数，若非泛型类型，返回 None，
            当 key 为字符串时，获得属性 property 的 type_hint
        """

        path = as_path(path)
        obj = self
        pos = 0
        for idx, p in enumerate(path):
            pos = idx
            if p is Path.tags.parent:
                obj = obj._parent
            elif p is Path.tags.root:
                obj = obj._root
            elif p is Path.tags.current:
                continue
            else:
                break

        path = path[pos:]

        tp_hint = getattr(obj, "__orig_class__", self.__class__)

        for key in path:
            if tp_hint is None:
                break
            elif isinstance(key, str):
                if typing.get_origin(tp_hint) is None:
                    tp_hint = typing.get_type_hints(tp_hint).get(key, None)
                else:
                    tp_hint = None
            else:
                tmp = get_args(tp_hint)
                if len(tmp) == 0:
                    tp_hint = None
                else:
                    tp_hint = tmp[-1]

        # logger.debug((path, tp_hint))

        return tp_hint

    def _as_child(self, value,  key, *args, entry: Entry | None = None,
                  type_hint: typing.Type = None,
                  default_value=_not_found_,
                  getter: typing.Callable | None = None,
                  force=True,  # 若type_hint为 None，强制 HTree
                  parent=None,  **kwargs) -> HTree[_T] | _T:

        if value is _not_found_ and entry is None:
            return _not_found_

        if parent is None:
            parent = self

        if default_value is _not_found_ or isinstance(default_value, collections.abc.Mapping):
            if isinstance(key, str) and isinstance(self._default_value, collections.abc.Mapping):
                s_default_value = deepcopy(self._default_value.get(key, _not_found_))

            else:
                s_default_value = deepcopy(self._default_value)

            default_value = merge_tree_recursive(s_default_value, default_value)

        if type_hint is None:
            type_hint = self._type_hint(key if key is not None else 0)

        if type_hint is None and force:
            type_hint = HTree[_T]

        metadata = kwargs.pop("metadata", {})

        if isinstance(key, str) and hasattr(self.__class__, key):
            metadata = merge_tree_recursive(metadata, getattr(getattr(self.__class__, key), "metadata", metadata))

        kwargs["metadata"] = metadata

        if value is not _not_found_:
            pass

        elif entry is None:
            value = default_value
            default_value = _not_found_

        elif not issubclass(get_origin(type_hint), HTree):
            value = entry.__value__
            entry = None
            if value is _not_found_:
                value = default_value
                default_value = _not_found_

        if not isinstance_generic(value, type_hint) and getter is not None:
            # if cache is not _not_found_ and cache is not None:
            #     logger.warning(f"Ignore {cache}")
            try:
                tmp = getter(self)
            except Exception as error:
                raise RuntimeError(f"{self.__class__} id={key}: 'getter' failed!") from error
            else:
                value = tmp

        if isinstance_generic(value, type_hint):
            pass

        elif type_hint in array_type:
            if isinstance(value, (list)) or isinstance(value, array_type):
                pass

        elif issubclass(get_origin(type_hint), HTree):
            value = type_hint(value, entry=entry, parent=parent, *args,
                              default_value=default_value,  **kwargs)

        elif not force and isinstance(value, HTree):
            value = value.__value__

        elif not force and isinstance(value, Entry):
            value = value.__value__

        else:
            if value is _not_found_ and entry is not None:
                value = entry.__value__

            if type_hint is not None:
                value = type_convert(value, type_hint=type_hint, **kwargs)

        return value

    def _get(self, query: PathLike = None,  *args, type_hint=None, **kwargs) -> HTree[_T] | _T:
        """ 获取子节点  """

        value = _not_found_

        if query is None:  # get value from self._entry and update cache
            return self

        elif query is Path.tags.current:
            value = self

        elif query is Path.tags.parent:
            value = self._parent
            if hasattr(value, "_identifier"):
                value = value._parent

        elif query is Path.tags.next:
            raise NotImplementedError(f"TODO: operator 'next'!")
            # value = self._parent.next(self)

        elif query is Path.tags.root:
            value = self._root

        elif isinstance(query, (int, slice, tuple, Query)):
            if type_hint in numeric_type:
                value = self._get_as_array(query, type_hint=type_hint, *args, **kwargs)
            else:
                value = self._get_as_list(query, *args, type_hint=type_hint,  **kwargs)

        elif isinstance(query, str):
            value = self._get_as_dict(query, *args,  type_hint=type_hint, **kwargs)

        elif isinstance(query, set):  # compound
            value = NamedDict(cache={k: self._get_as_dict(
                k, type_hint=type_hint, *args,  **kwargs) for k in query})

        else:
            raise NotImplementedError(f"TODO: {type(query)}")

        return value  # type:ignore

    def _get_as_array(self, query, *args, **kwargs) -> NumericType:

        if self._cache is _not_found_:
            self._cache = self._entry.__value__  # type:ignore

        if isinstance(self._cache, array_type) or isinstance(self._cache, collections.abc.Sequence):
            return self._cache[query]

        elif self._cache is _not_found_:
            return kwargs.get("default_value", _not_found_)  # type:ignore

        else:
            raise RuntimeError(f"{self._cache}")

    def _get_as_dict(self, key: str,  *args, **kwargs) -> HTree[_T] | _T:

        cache = _not_found_

        if isinstance(self._cache, collections.abc.Mapping):
            cache = self._cache.get(key, _not_found_)

        if self._entry is not None:
            entry = self._entry.child(key)
        else:
            entry = None

        value = self._as_child(cache, key, *args, entry=entry, **kwargs)

        if self._cache is _not_found_ or self._cache is None:
            self._cache = {}

        self._cache[key] = value

        return value

    def _get_as_list(self, key: PathLike,  *args, default_value=_not_found_, **kwargs) -> HTree[_T] | _T:

        if isinstance(key, (Query, dict)):
            raise NotImplementedError(f"TODO:")
            # cache = QueryResult(self, key, *args, **kwargs)
            # entry = None
            # key = None

        elif isinstance(key, int):
            if isinstance(self._cache, list) and key < len(self._cache):
                cache = self._cache[key]
                if isinstance(key, int) and key < 0:
                    key = len(self._cache)+key
            else:
                cache = _not_found_

            entry = self._entry.child(key)
        elif isinstance(key, slice):
            start = key.start or 0
            stop = key.stop
            step = key.step or 1

            if isinstance(self._cache, list):
                if stop is not None and stop < 0 and start >= len(self._cache):
                    raise NotImplementedError()
                else:
                    cache = self._cache[slice(start, stop, step)]
            else:
                cache = _not_found_

            entry = self._entry.child(key)

        elif self._cache is _not_found_ or self._cache is None:
            entry = self._entry.child(key)
            cache = None

        else:
            raise RuntimeError((key, self._cache, self._entry))

        if default_value is _not_found_:
            default_value = self._default_value

        value = self._as_child(cache, key, *args, entry=entry,
                               parent=self._parent, default_value=default_value, **kwargs)

        if isinstance(key, int):
            if self._cache is _not_found_:
                self._cache = []
            elif not isinstance(self._cache, list):
                raise ValueError(self._cache)

            if key >= len(self._cache):
                self._cache.extend([_not_found_] * (key - len(self._cache) + 1))

        if isinstance(key, int) and (key < 0 or key >= len(self._cache)):
            raise IndexError(f"Out of range  {key} > {len(self._cache)}")

        if isinstance(key, int) and key >= 0:
            self._cache[key] = value

        return value

    def _query(self,  path: PathLike,   *args,  **kwargs) -> HTree[_T] | _T:
        if self._cache is not _not_found_:
            return as_path(path).fetch(self._cache, *args, **kwargs)
        else:
            return self._entry.child(path).fetch(*args, **kwargs)

    def _insert(self, path: PathLike,  *args, **kwargs):
        tmp = {"_": self._cache}
        as_path(["_"]+as_path(path)[:]).insert(tmp,  *args, **kwargs)
        self._cache = tmp["_"]
        return self

    def _update(self, path: PathLike,  *args, **kwargs):
        tmp = {"_": self._cache}
        as_path(path).prepend(["_"]).update(tmp,   *args, **kwargs)
        self._cache = tmp["_"]
        return self

    def _remove(self, path: PathLike,  *args, **kwargs) -> None:
        self.update(path, _not_found_)
        self._entry.child(path).remove(*args, **kwargs)

    @deprecated
    def _find_next(self, query: PathLike, start: int | None,   default_value=_not_found_, **kwargs) -> typing.Tuple[typing.Any, int | None]:

        if query is None:
            query = slice(None)

        cache = None
        entry = None
        next_id = start
        if isinstance(query, slice):

            start_q = query.start or 0
            stop = query.stop
            step = query.step
            if start is None:
                start = start_q
            else:
                start = int((start-start_q)/step)*step

            next_id = start+step
            if isinstance(self._cache, list) and start < len(self._cache):
                cache = self._cache[start]
            entry = self._entry.child(start)

        elif isinstance(query, Query):
            pass

        if start is not None:
            return self._as_child(cache, start,  entry=entry, default_value=default_value, **kwargs), next_id
        else:
            return None, None


def as_htree(obj, *args, **kwargs):
    if isinstance(obj, HTree):
        return obj
    else:
        return HTree(obj, *args, **kwargs)


Node = HTree


class Container(HTree[_T]):

    pass


class Dict(Container[_T]):

    def __iter__(self) -> typing.Generator[str, None, None]:
        """ 遍历 children """
        for k in self.children():
            yield k

    def items(self): yield from self.children()

    def __contains__(self, key: str) -> bool:
        return (isinstance(self._cache, collections.abc.Mapping) and key in self._cache) or self._entry.child(key).exists


class List(Container[_T]):
    def __init__(self, cache: typing.Any = None, *args, **kwargs) -> None:
        if cache is _not_found_:
            pass
        elif not isinstance(cache, collections.abc.Sequence):
            cache = [cache]
        super().__init__(cache, *args, **kwargs)

    def __iter__(self) -> typing.Generator[ _T, None, None]:
        """ 遍历 children """
        for v in self.children():
            yield v

    def __getitem__(self, path) -> _T: return super().__getitem__(path)


# class QueryResult(HTree[_T]):
#     """ Handle the result of query    """

#     def __init__(self, query: PathLike, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self._path = as_path(query)

#     def __getattr__(self, name: str): return self._get(name)

#     def _get(self, query: str | int | slice | dict, *args, **kwargs):
#         default_value = kwargs.pop("default_value", _not_found_)
#         _VT = get_args(self.__orig_class__)[0]
#         if isinstance(query, str):

#             if default_value is _not_found_ and isinstance(self._default_value, dict):
#                 default_value = self._default_value.get(query, _not_found_)
#             tp = get_type_hint(_VT, query)

#             return QueryResult[tp](self._path.append(query), *args, default_value=default_value, **kwargs)
#         else:
#             return QueryResult[_VT](self._path.append(query), *args, default_value=default_value, **kwargs)

#     @property
#     def __value__(self) -> typing.Any:
#         value = super()._query(self._path)
#         if isinstance(value, list):
#             value = functools.reduce(self._default_reducer, value)
#         return value

#     def __call__(self, *args, **kwargs) -> typing.Any:

#         value = super()._query(self._path, op=Path.tags.call, *args, **kwargs)

#         if isinstance(value, list):
#             value = functools.reduce(self._default_reducer, value)

#         return value

#     def __iter__(self) -> typing.Generator[typing.Tuple[str, _T | HTree[_T]] | _T | HTree[_T], None, None]:
#         raise NotImplementedError(f"TODO:")

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

#     def children(self) -> typing.Generator[_T | HTree[_T], None, None]:
#         """ 遍历 children """
#         cache = self._cache if self._cache is not _not_found_ else self._default_value

#         if not isinstance(cache, list) or len(cache) == 0:
#             yield from super().children()

#         else:
#             for idx, value in enumerate(cache):
#                 if isinstance(value, (dict, Dict)):
#                     id = value.get(self._identifier, None)
#                 else:
#                     id = None
#                 if id is not None:
#                     entry = self._entry.child({f"@{self._identifier}": id})
#                 else:
#                     entry = None

#                 yield self._as_child(value, idx, entry=entry)


# class AoS(List[_T]):
#     """
#         Array of structure
#     """

#     def __init__(self, *args, identifier: str | None = None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._identifier = identifier
#         if self._identifier is None:
#             self._identifier = self.__metadata__.get("identifier", None)

#     def _get(self, query: PathLike,  **kwargs) -> HTree[_T] | _T | QueryResult[_T]:

#         if isinstance(query, int):
#             return super()._get(query)

#         elif isinstance(query, str):
#             query = {f"@{self._identifier}": query}

#         elif not isinstance(query, (slice, dict)):
#             raise TypeError(f"{type(query)}")

#         default_value = kwargs.pop("default_value", self._default_value)

#         tp = self._type_hint(0)

#         return QueryResult[tp](query, self._cache, entry=self._entry, default_value=default_value, parent=self._parent, **kwargs)

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
