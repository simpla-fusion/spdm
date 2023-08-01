from __future__ import annotations

import collections.abc
import typing
from copy import copy

from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (ArrayType, HTreeLike, NumericType, array_type,
                            as_array, as_value, get_args, get_origin,
                            isinstance_generic, numeric_type, serialize,
                            type_convert)
from .Entry import Entry, as_entry
from .Path import Path, PathLike, as_path, Query, QueryLike, as_query

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

    def __init__(self, cache: typing.Any = None, entry: HTreeLike | Entry = None,
                 parent: HTree | None = None,
                 **kwargs) -> None:

        default_value = _not_found_

        # if isinstance(entry, dict):
        #     default_value = merge_tree_recursive(default_value, (entry.pop("$default_value", {})))

        if isinstance(cache, dict):
            default_value = merge_tree_recursive(default_value, (cache.pop("$default_value", _not_found_)))

        default_value = merge_tree_recursive(default_value, kwargs.pop("default_value", _not_found_))

        if cache is None or cache is _undefined_:
            cache = _not_found_
        self._cache = cache
        self._entry = as_entry(entry)
        self._default_value = default_value
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
        return as_value(self._cache)

    def __array__(self) -> ArrayType: return as_array(self.__value__)

    def _repr_svg_(self) -> str:
        from ..views.View import display
        return display(self, output="svg")

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

    def __getitem__(self, path) -> HTree[_T] | _T: return self.get(path, force=True)

    def __setitem__(self, path, value) -> None: self._update(path, value)

    def __delitem__(self, path) -> None: return self._remove(path)

    def __contains__(self, key) -> bool: return self._query([key], Path.tags.exists)  # type:ignore

    def __len__(self) -> int: return self._query([], Path.tags.count)  # type:ignore

    def __iter__(self) -> typing.Generator[_T | HTree[_T], None, None]:
        """ 遍历 children """

        next_id = []

        while True:

            value, next_id = self._find_next(None, *next_id, parent=self._parent,
                                             default_value=self._default_value)
            if next_id is None or len(next_id) == 0:
                break
            yield value

    def __equal__(self, other) -> bool: return self._query([], Path.tags.equal, other)  # type:ignore

    def insert(self,  *args, **kwargs): return self._insert([], *args, **kwargs)

    def update(self, *args, **kwargs): return self._update([], *args, **kwargs)

    def get(self, path: Path | PathLike,  default_value=_not_found_, *args,   force=False, **kwargs) -> HTree[_T] | _T:

        path = as_path(path)
        length = len(path)

        if length == 0:
            return self

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

    def _as_child(self, cache,  key, *args, entry: Entry | None = None,
                  type_hint: typing.Type = None,
                  default_value=_not_found_,
                  getter: typing.Callable | None = None,
                  force=True,  # 若type_hint为 None，强制 HTree
                  parent=None,  **kwargs) -> HTree[_T] | _T:

        if cache is _not_found_ and entry is None:
            return _not_found_

        if parent is None:
            parent = self

        if default_value is _not_found_ or isinstance(default_value, collections.abc.Mapping):
            if isinstance(key, str) and isinstance(self._default_value, collections.abc.Mapping):
                s_default_value = self._default_value.get(key, _not_found_)

            else:
                s_default_value = self._default_value

            default_value = merge_tree_recursive(s_default_value, default_value)

        if cache is not _not_found_:
            pass
        elif not force and entry is not None:
            cache = entry.__value__
            entry = None
        else:
            cache = default_value
            default_value = _not_found_

        if type_hint is None:
            type_hint = self._type_hint(key if key is not None else 0)

        if type_hint is None and force:
            type_hint = HTree[_T]

        if not isinstance_generic(cache, type_hint) and getter is not None:
            # if cache is not _not_found_ and cache is not None:
            #     logger.warning(f"Ignore {cache}")
            try:
                tmp = getter(self)
            except Exception as error:
                raise RuntimeError(f"id={key}:'getter' failed!") from error
            else:
                cache = tmp

        if isinstance_generic(cache, type_hint):
            value = cache

        elif issubclass(get_origin(type_hint), HTree):
            value = type_hint(cache, entry=entry, parent=parent, *args, default_value=default_value, **kwargs)

        elif not force and isinstance(cache, HTree):
            value = cache.__value__

        elif not force and isinstance(cache, Entry):
            value = cache.__value__

        else:
            value = cache if cache is not _not_found_ else entry.__value__
            if type_hint is not None:
                value = type_convert(value, type_hint=type_hint, **kwargs)

        return value

    def _get(self, query: PathLike = None, type_hint=None,  *args, **kwargs) -> HTree[_T]:
        """ 获取子节点  """

        value = _not_found_

        if query is None:  # get value from self._entry and update cache
            return self

        elif query is Path.tags.current:
            value = self

        elif query is Path.tags.parent:
            value = self._parent

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

        if isinstance(key, (Query, dict,  slice)):
            cache = None
            entry = QueryEntry(self, key, *args, **kwargs)
            key = None

        elif isinstance(key, int):
            if isinstance(self._cache, list):
                cache = self._cache[key]
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

            self._cache[key] = value

        return value

    def _query(self,  path: PathLike,   *args,  **kwargs) -> HTree[_T] | _T:
        if self._cache is not _not_found_:
            return as_path(path).fetch(self._cache, *args, **kwargs)
        else:
            return self._entry.child(path).query(*args, **kwargs)

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

    def _find_next(self, query: PathLike = None, *starts: int | None,   default_value=_not_found_, **kwargs) -> typing.Tuple[typing.Any, typing.List[int | None]]:

        if query is None:
            query = slice(None)

        cache, pos = as_path(query).find_next(self._cache, *starts)

        if pos is None:
            cache = _not_found_
            entry, pos = self._entry.child(query).find_next(*starts)
        else:
            entry = self._entry.child(pos)

        if pos is not None:
            return self._as_child(cache, pos,  entry=entry, default_value=default_value, **kwargs), pos
        else:
            return None, None


def as_htree(obj, *args, **kwargs):
    if isinstance(obj, HTree):
        return obj
    else:
        return HTree(obj, *args, **kwargs)


Node = HTree


class Container(HTree[_T]):

    def __getitem__(self, path: PathLike) -> HTree[_T] | _T: return super().__getitem__(path)


class Dict(Container[_T]):

    def __getitem__(self, path) -> HTree[_T] | _T: return super().__getitem__(path)

    def __contains__(self, key: str) -> bool:
        return (isinstance(self._cache, collections.abc.Mapping) and key in self._cache) or self._entry.child(key).exists


class List(Container[_T]):
    def __init__(self, cache: typing.Any = None, *args, **kwargs) -> None:
        if cache is _not_found_:
            pass
        elif not isinstance(cache, collections.abc.Sequence):
            cache = [cache]
        super().__init__(cache, *args, **kwargs)

    def __getitem__(self, path) -> HTree[_T] | _T: return super().__getitem__(path)


class NamedDict(HTree[_T]):
    """ Proxy to access named dict """

    def __getattr__(self, name: str) -> typing.Any: return self._get(name)


class QueryEntry(Entry):
    """ Handle the result of query    """

    def __init__(self, target: typing.Any, query: QueryLike,  *args,
                 reducer: typing.Callable[..., typing.Any] | None = None, **kwargs) -> None:
        super().__init__(target, as_query(query), *args, **kwargs)
        self._reducer = reducer if reducer is not None else QueryEntry._default_reducer

    def __copy_from__(self, other: QueryEntry) -> QueryEntry:
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
        res = [(e.get(*args, default_value=_not_found_, **kwargs) if isinstance(e, Entry) else e)
               for e in self._foreach()]

        res = [e for e in res if e is not _not_found_]

        if len(res) == 0:
            res = [default_value]

        return res

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
            if next_id is None or len(next_id) == 0:
                break
            yield entry

    @ property
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


class AoS(List[_T]):
    """
        Array of structure
    """

    def __init__(self, *args, identifier: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._identifier = identifier
        if self._identifier is None:
            self._identifier = self.__metadata__.get("identifier", "id")

    def __getitem__(self, path) -> HTree[_T] | _T: return super().__getitem__(path)

    def __iter__(self) -> typing.Generator[_T | HTree[_T], None, None]:
        if not isinstance(self._default_value, collections.abc.Sequence):
            yield from super().__iter__()
        else:
            for d in self._default_value:
                yield self[d.get(self._identifier)]

    def _get(self, query:   PathLike,  *args, default_value=_not_found_, type_hint=None, **kwargs) -> HTree[_T] | _T:

        if default_value is _not_found_:
            default_value = self._default_value

        if isinstance(query, str):
            query = Query({f"@{self._identifier}": query})

        return super()._get_as_list(query, *args, type_hint=type_hint, default_value=default_value, **kwargs)
