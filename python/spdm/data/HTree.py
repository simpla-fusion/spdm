from __future__ import annotations

import collections.abc
import inspect
import typing
from copy import copy, deepcopy
from functools import reduce

from ..utils.logger import logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (ArrayType, HNodeLike, HTreeLike, NumericType,
                            PrimaryType, array_type, as_array, as_value,
                            get_args, get_origin, isinstance_generic,
                            numeric_type, primary_type, serialize,
                            type_convert)
from .Entry import Entry, as_entry
from .Expression import Expression
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

    def __init__(self, cache: typing.Any = None, entry: HTreeLike | Entry = None,
                 parent: HTree | None = None,
                 **kwargs) -> None:

        default_value = {}

        # if isinstance(entry, dict):
        #     default_value = merge_tree_recursive(default_value, (entry.pop("$default_value", {})))

        if isinstance(cache, dict):
            default_value = merge_tree_recursive(default_value, (cache.pop("$default_value", {})))

        default_value = merge_tree_recursive(default_value, kwargs.pop("default_value", {}))

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
        other._cache = copy(self._cache)
        other._entry = copy(self._entry)
        other._parent = self._parent
        other._metadata = copy(self._metadata)
        other._default_value = copy(self._default_value)

        return other

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

    def __getitem__(self, path) -> HTree[_T] | _T: return self.get(path, force=False)

    def __setitem__(self, path, value) -> None: self._update(path, value)

    def __delitem__(self, path) -> None: return self._remove(path)

    def __contains__(self, key) -> bool: return self._query([key], Path.tags.exists)  # type:ignore

    def __len__(self) -> int: return self._query([], Path.tags.count)  # type:ignore

    def __iter__(self) -> typing.Generator[_T | HTree[_T], None, None]:
        """ 遍历 children """

        next_id: PathLike = None

        while True:

            value, next_id = self._find_next(start=next_id, parent=self._parent,
                                             default_value=self._default_value)
            if next_id is None:
                break

            yield value

    def __equal__(self, other) -> bool: return self._query([], Path.tags.equal, other)  # type:ignore

    def insert(self,  *args, **kwargs): return self._insert([], *args, **kwargs)

    def update(self, *args, **kwargs): return self._update([], *args, **kwargs)

    def get(self, path: Path | PathLike,  default_value=_undefined_, *args, force=True, **kwargs) -> HTree[_T] | _T:

        path = as_path(path)
        length = len(path)

        if length == 0:
            return self

        obj = self

        pos = -1

        for idx, p in enumerate(path[:-1]):

            if isinstance(obj, HTree):
                tmp = obj._get(p, default_value=_not_found_)
                pos = idx
            else:
                tmp = Path(path[idx:]).query(obj, default_value=_not_found_)
                pos = len(path)
                break

            if tmp is _not_found_ or pos >= length:
                break
            else:
                obj = tmp

        if isinstance(obj, HTree) and pos == length-2:
            obj = obj._get(path[-1], *args, default_value=default_value, **kwargs)

        if obj is _not_found_ or obj is _undefined_:
            obj = default_value

        if obj is _undefined_ and pos <= len(path):
            raise KeyError(f"{path[:pos+1]} not found")

        if isinstance(obj, HTree) and force:
            tp = obj._type_hint()
            if tp is None or tp == HTree[_T] or tp in primary_type:
                obj = obj.__value__
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
                    tp_hint = HTree[_T]
                else:
                    tp_hint = tmp[-1]

        # logger.debug((path, tp_hint))

        return tp_hint

    def _as_child(self, cache,  id, *args, entry=None,
                  type_hint: typing.Type = None,
                  default_value=_not_found_,
                  getter: typing.Callable = None,
                  parent=None,  **kwargs):

        if entry is None:
            entry = self._entry.child(id)

        if parent is None:
            parent = self

        if type_hint is None:
            type_hint = self._type_hint(id) or HTree[_T]

        if default_value is not _not_found_:
            pass

        elif isinstance(id, str) and isinstance(self._default_value, collections.abc.Mapping):
            default_value = self._default_value.get(id, _not_found_)

        else:
            default_value = self._default_value

        if cache is _not_found_:
            cache = default_value
            default_value = _not_found_

        # if type_hint is None and isinstance(cache, primary_type):
        #     value = cache
        # elif type_hint is None and not isinstance(cache, primary_type):
        #     value = HTree[_T](cache, entry, parent=parent, *args, **kwargs)
        # el
        if isinstance_generic(cache, type_hint):
            value = cache
        elif getter is not None:
            # logger.warning(f"Ignore {cache}")
            value = getter(self)

        elif issubclass(get_origin(type_hint), HTree):
            value = type_hint(cache, entry=entry, parent=parent, *args, default_value=default_value, **kwargs)

        else:
            value = type_convert(cache if cache is not _not_found_ else entry.__value__, type_hint=type_hint, **kwargs)

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

        elif isinstance(query, (int, slice, tuple)) and type_hint in numeric_type:
            value = self._get_as_array(query, type_hint=type_hint, *args, **kwargs)

        elif isinstance(query, str):
            value = self._get_as_dict(query, *args,  type_hint=type_hint, **kwargs)

        elif isinstance(query, (slice, int)):
            value = self._get_as_list(query, *args, type_hint=type_hint,  **kwargs)

        elif isinstance(query,  dict):  # as query return QueryResult
            value = self._get_as_htable(query, *args, type_hint=type_hint,  **kwargs)

        elif isinstance(query, set):  # compound
            value = NamedDict(cache={k: self._get_as_dict(
                k, type_hint=type_hint, *args,  **kwargs) for k in query})

        else:
            raise NotImplementedError(f"TODO: {type(query)}")

        return value  # type:ignore

    def _get_as_htable(self, query: dict, *args,  **kwargs) -> HTree[_T] | _T | QueryResult:
        """ Hierarchical Table(htable): searchable list """

        if not all([k.startswith("@") for k in query.keys()]):
            return QueryResult(self, query, *args, **kwargs)
        else:
            pos = self._entry.query(Path.tags.search, query)
            if pos is not None and pos is not _not_found_:
                return self._get_as_list(pos, *args,  **kwargs)
            else:
                return _not_found_

    def _get_as_array(self, query, *args, **kwargs) -> NumericType:
        if self._cache is _not_found_:
            self._cache = copy(self._entry.__value__)  # type:ignore

        if isinstance(self._cache, array_type) or isinstance(self._cache, collections.abc.Sequence):
            return self._cache[query]

        elif self._cache is _not_found_:
            return kwargs.get("default_value", _not_found_)  # type:ignore

        else:
            raise RuntimeError(f"{self._cache}")
        pass

    def _get_as_dict(self, key: str,  *args, **kwargs) -> HTree[_T] | _T:

        cache = _not_found_

        if isinstance(self._cache, collections.abc.Mapping):
            cache = self._cache.get(key, _not_found_)

        value = self._as_child(cache, key, *args, **kwargs)

        if value is not _not_found_:
            if self._cache is _not_found_ or self._cache is None:
                self._cache = {}
            self._cache[key] = value

        return value

    def _get_as_list(self, key: int | slice,  *args, default_value=_not_found_, **kwargs) -> HTree[_T] | _T | QueryResult:

        if isinstance(key, slice):
            return QueryResult(self, key, *args, **kwargs)

        cache = None

        if isinstance(self._cache, collections.abc.Sequence):
            cache = self._cache[key]

        elif self._cache is not _not_found_:
            raise RuntimeError(self._cache)

        if default_value is _not_found_ or default_value is _undefined_:
            default_value = self._default_value

        value = self._as_child(cache, key, *args, parent=self._parent, default_value=default_value, **kwargs)

        if value is not _not_found_ and isinstance(key, int):
            if isinstance(self._cache, list):
                pass
            elif self._entry.exists and self._entry.count > 0:
                self._cache = [_not_found_] * self._entry.count
            else:
                self._cache = []
            self._cache[key] = value

        return value

    def _query(self,  path: PathLike,   *args,  **kwargs) -> HTree[_T] | _T:
        if self._cache is not _not_found_:
            return as_path(path).query(self._cache, *args, **kwargs)
        else:
            return self._entry.child(path).query(*args, **kwargs)

    def _insert(self, path: PathLike,  *args, **kwargs):
        tmp = {"_": self._cache}
        as_path(["_"]+as_path(path)[:]).insert(tmp,  *args, **kwargs)
        self._cache = tmp["_"]
        return self

    def _update(self, path: PathLike,  *args, **kwargs):
        tmp = {"_": self._cache}
        as_path(["_"]+as_path(path)[:]).update(tmp,   *args, **kwargs)
        self._cache = tmp["_"]
        return self

    def _remove(self, path: PathLike,  *args, **kwargs) -> None:
        self.update(path, _not_found_)
        self._entry.child(path).remove(*args, **kwargs)

    def _find_next(self, query: PathLike = None, start: PathLike = None, *args, default_value=_not_found_, **kwargs) -> typing.Tuple[_T | HTree[_T], PathLike]:

        cache, pos = as_path(query).find_next(self._cache, start=start, *args, **kwargs)

        if pos is None:
            cache = _not_found_
            entry, pos = self._entry.child(query).find_next(start=start, *args, **kwargs)
        else:
            entry = self._entry.child(pos)

        if pos is not None:
            return self._as_child(cache, pos, *args, entry=entry, default_value=default_value,  **kwargs), pos
        else:
            return None, None


def as_htree(obj, *args, **kwargs):
    if isinstance(obj, HTree):
        return obj
    else:
        return HTree(obj, *args, **kwargs)


Node = HTree


class Container(HTree[_T]):

    def __getitem__(self, path: PathLike) -> HTree[_T] | _T: return self.get(path, force=False)


class Dict(Container[_T]):

    def __getitem__(self, path) -> HTree[_T] | _T: return self.get(path, force=False)

    def __contains__(self, key: str) -> bool:
        return (isinstance(self._cache, collections.abc.Mapping) and key in self._cache) or self._entry.child(key).exists


class List(Container[_T]):

    def __getitem__(self, path) -> HTree[_T] | _T: return self.get(path, force=False)


class NamedDict(HTree[_T]):
    """ Proxy to access named dict """

    def __getattr__(self, name: str) -> typing.Any: return self._get(name)


class QueryResult(Expression):
    """ Handle the result of query    """

    def __init__(self, target: HTree, query: PathLike,  *args, suffix: PathLike | Path = None,
                 reducer: typing.Callable[..., HTreeLike] | None = None, **kwargs) -> None:
        # super().__init__(*args, **kwargs)
        self._query_cmd = query
        self._suffix = as_path(suffix)
        self._target = as_htree(target)
        self._reducer = reducer if reducer is not None else QueryResult._default_reducer

    def __copy__(self) -> QueryResult:
        return QueryResult(self._target, self._query_cmd, suffix=copy(self._suffix), reducer=self._reducer)

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

    def _foreach(self, suffix=None) -> typing.Generator[HTreeLike | HTree, None, None]:

        if suffix is None:
            suffix = self._suffix
        else:
            suffix = as_path(suffix)
        start = None

        while True:
            value, start = self._target._find_next(self._query_cmd, start=start)

            if start is None:
                break

            elif len(suffix) == 0:
                pass

            elif isinstance(value, HTree):
                value = value[suffix]

            else:
                value = suffix.query(value, default_value=_not_found_)

            yield value

    @ property
    def __value__(self) -> typing.List[HTreeLike]:
        value = [as_value(v) for v in self._foreach() if v is not _not_found_]
        if len(value) == 0 or isinstance(value[0], collections.abc.Mapping)\
                and self._target._default_value is not _not_found_ and len(self._suffix) > 0:
            default_value = self._suffix.collapse().query(self._target._default_value)
            value = [default_value]+value
        return value

    def __array__(self) -> ArrayType: return as_array(self.__reduce__())

    def __reduce__(self) -> HTreeLike: return reduce(self._reducer,  self.__value__)

    @ staticmethod
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

    def __call__(self, *args, **kwargs) -> HTreeLike:
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

    def __getitem__(self, path) -> HTree[_T] | _T:
        path = as_path(path)

        if len(path) > 0 and isinstance(path[0], str):
            path[0] = {f"@{self._identifier}": path[0]}

        return super().__getitem__(path)
