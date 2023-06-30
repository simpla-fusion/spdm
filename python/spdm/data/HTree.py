from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import pprint
import typing
from copy import copy
from enum import Enum

import numpy as np

from ..utils.logger import logger
from ..utils.numeric import is_close
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import (ArrayType, NumericType, PrimaryType, array_type,
                            as_array, get_args, get_origin, numeric_type,
                            get_type_hint, isinstance_generic, primary_type,
                            serialize, type_convert)
from .Entry import Entry, as_entry
from .Path import Path, PathLike, as_path, path_like

_T = typing.TypeVar("_T")

HTreeLike = dict | list | int | str | float | bool | np.ndarray | None | Entry


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

    def __init__(self, data: HTreeLike | Entry = None,  cache: typing.Any = None, parent: HTree | None = None, **kwargs) -> None:

        self._entry = as_entry(data)
        # self._cache = cache
        self._parent = parent
        self._metadata = Path().update(kwargs.pop("metadata", {}), kwargs)

    def __copy__(self) -> HTree[_T]:
        other: HTree = self.__class__.__new__(getattr(self, "__orig_class__", self.__class__))
        other._entry = copy(self._entry)
        other._metadata = copy(self._metadata)
        other._parent = self._parent
        other._cache = copy(other._cache)
        return other

    def __serialize__(self) -> typing.Any: return serialize(self.__value__)

    @classmethod
    def __deserialize__(cls, *args, **kwargs) -> HTree: return cls(*args, **kwargs)

    def __str__(self) -> str: return f"<{self.__class__.__name__} />"

    def __array__(self) -> ArrayType: return as_array(self.__value__)

    def _repr_svg_(self) -> str:
        from ..views.View import display
        return display(self, output="svg")

    @property
    def __name__(self) -> str: return self._metadata.get("name", "unamed")

    @property
    def __metadata__(self) -> dict: return self._metadata

    def __type_hint__(self, key: PathLike = None) -> typing.Type | None:
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

    def __getitem__(self, path) -> HTree[_T] | _T | PrimaryType: return self.get(path)

    def __setitem__(self, path, value) -> None: self._query(path, Path.tags.insert, value)

    def __delitem__(self, path) -> bool: return self._query(path, Path.tags.remove)

    def __contains__(self, path) -> bool: return self._query(path, Path.tags.exists) > 0

    def __len__(self) -> int: return self._query([], Path.tags.count) > 0

    def __iter__(self) -> typing.Generator[typing.Any, None, None]: yield from self._find(slice(None))

    # def __equal__(self, other) -> bool: return self._entry.__equal__(other) if self._entry is not None else (other is None)

    @property
    def __value__(self) -> typing.Any:

        if isinstance(self._cache, dict):
            raise NotImplementedError(f"TODO: merge cache and entry")

        elif self._cache is not None and self._cache is not _not_found_:
            return self._cache

        else:
            default_value = self._metadata.get("default_value", _not_found_)
            tmp = self._entry.get(default_value=_not_found_)
            tmp = merge_tree_recursive(default_value, tmp)
            if tmp is not _not_found_:
                self._cache = tmp
            return tmp

    def get(self, path: Path | PathLike, **kwargs) -> typing.Any:
        path = as_path(path)
        target = self
        for idx, p in enumerate(path):
            if target is _not_found_:
                break
            elif not isinstance(target, HTree):
                target = Path(path[idx:]).query(target)
                break
            else:
                target = target._get(p, default_value=_not_found_)
        else:
            if target is _not_found_:
                target = kwargs.get("default_value", _not_found_)

        return target

    def _get(self, key: PathLike,  **kwargs) -> _T | ListProxy[_T] | typing.Dict[str, _T] | ArrayType:
        """ 获取子节点  """

        type_hint = self.__type_hint__()

        value = _not_found_

        if isinstance(key, (int, slice, tuple)) and type_hint in numeric_type:
            if self._cache is None or len(self._cache) == 0:
                self._cache = self._entry.__value__  # type:ignore

            if isinstance(self._cache, array_type) or isinstance(self._cache, collections.abc.Sequence):
                value = self._cache[key]

            elif self._cache is None or self._cache is _not_found_:
                return kwargs.get("default_value", _not_found_)  # type:ignore
            else:
                raise RuntimeError(f"{self._cache}")

        elif isinstance(key, str):
            value = self._get_by_name(key, type_hint=type_hint, **kwargs)

        elif isinstance(key, set):
            value = {k: self._get_by_name(k, type_hint=type_hint, **kwargs) for k in key}

        elif isinstance(key, int):
            value = self._get_by_index(key, type_hint=type_hint, **kwargs)

        elif key is not None:
            value = ListProxy[_T](self._entry.child(key) if self._entry is not None else None,
                                  cache=self._cache,
                                  metadata=self._metadata, parent=self._parent)

        else:
            value = self.__value__

        return value  # type:ignore

    def _get_by_name(self, key: str,  default_value=_not_found_, type_hint=None, getter=None) -> _T:

        type_hint = self.__type_hint__(key) or type_hint or HTree[_T]

        value = _not_found_

        # if isinstance(self._cache, collections.abc.Mapping):
        #     value = self._cache.get(key, _not_found_)

        if value is _not_found_ and getter is not None:
            value = getter(self)

        if value is _not_found_ and isinstance(self._entry, Entry):
            value = self._entry.child(key)

        if default_value is _not_found_:
            default_value = self._metadata.get("default_value", {}).get(key, _not_found_)

        value = type_convert(value, type_hint=type_hint, default_value=default_value, parent=self)

        # if self._cache is None:
        #     self._cache = {}

        # elif not isinstance(self._cache, collections.abc.Mapping):
        #     raise TypeError(f"{type(key)}")

        # self._cache[key] = value

        return value

    def _get_by_index(self, key: int,  default_value=_not_found_, type_hint=None) -> _T:

        if type_hint is None:
            type_hint = self.__type_hint__() or HTree[_T]

        if default_value is _not_found_:
            default_value = self._metadata.get("default_value", {})

        if key < 0:
            if self._entry is not None:
                key += self._entry.count

        value = _not_found_

        # if isinstance(self._cache, collections.abc.Mapping):
        #     value = self._cache.get(key, _not_found_)

        if value is _not_found_ and isinstance(self._entry, Entry):
            value = self._entry.child(key)

        if not isinstance_generic(value, type_hint):
            value = type_convert(value, type_hint=type_hint, default_value=default_value, parent=self._parent)

        # if self._cache is None:
        #     self._cache = {}

        # self._cache[key] = value

        return value

    @property
    def _root(self) -> HTree | None:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while p._parent is not None and getattr(p, "ids_properties", None) is None:
            p = p._parent
        return p

    def _find(self, query, *args, **kwargs) -> typing.Generator[HTree, None, None]:
        entry = self._entry.child(query) if query is not None else self._entry

        for p, v in entry.find():
            yield self.as_child_deep(p, v, *args, **kwargs)

    def _as_child(self, key: PathLike,
                  value: typing.Any = _not_found_,
                  default_value: typing.Any = _undefined_,
                  type_hint: typing.Type = _not_found_,
                  parent: HTree = None, **kwargs) -> HTree[_T] | _T:
        """ 获取子节点   """
        if parent is None:
            parent = self

        if isinstance(key, set):
            return DictProxy[_T]({k: self.as_child(k, default_value=default_value, parent=parent, type_hint=type_hint, **kwargs) for k in key})

        elif isinstance(key, (slice, dict)):
            return ListProxy[_T](self._entry.child(key), default_value=default_value, parent=parent, type_hint=type_hint, **kwargs)

        elif isinstance(key, tuple):
            raise NotImplementedError(f"tuple {key}")
            # return ListProxy([self.as_child(k, default_value=default_value,   parent=parent, type_hint=type_hint, **kwargs) for k in key])

        elif isinstance(key, str):
            if value is _not_found_ or value is None:
                value = self._entry.child(key)

            if type_hint is None or type_hint is _not_found_:
                type_hint = self.__type_hint__(key)

            if default_value is None or default_value is _not_found_:
                if isinstance(self._default_value, collections.abc.Mapping):
                    default_value = self._default_value.get(key, _not_found_)
                else:
                    default_value = self._default_value

        else:
            type_hint = self.__type_hint__()

            default_value = self._default_value

            if (value is _not_found_ or value is None) and (key is not None and key is not _not_found_):
                value = self._entry.child(key)

    def _as_child_deep(self, path:  PathLike | Path | None = None, value=None,

                       default_value: typing.Any = _not_found_, **kwargs) -> HTree:
        """
            将 value 转换为 Node
            ----
            Parameters:
                - obj: 任意类型的数据
                - type_hint: 将obj转换到目标类型
                - path: child在obj中的路径
                - default_value: 默认值
                - kwargs: 其他参数
            若 obj 为 Node，调用 obj.get(path, **kwargs)
            当 type_hint 为 None 时，将 obj 转换为 entry
            当 type_hint 不为 None 时，根据 path 获得 obj 的子节点（as entry）， 根据 path 从 type_hint 中获取子类型，
        """

        path = Path(path)

        if len(path) == 0:
            return self

        parent = self._parent

        obj = self

        for idx, key in enumerate(path[:]):

            if obj is None or obj is _not_found_:
                break
            elif not isinstance(obj, HTree):
                obj = as_entry(obj).child(path[idx:])
                if isinstance(parent, HTree):
                    obj = parent.as_child(key, obj, default_value=default_value, **kwargs)
                else:
                    obj = HTree(obj,  default_value=default_value, **kwargs)

            if key == Path.tags.root:
                obj = obj._root
                parent = None
                continue
            elif key is Path.tags.parent:
                obj = obj._parent
                parent = None
                continue
            # elif key == "*":
            #     key = slice(None)

            parent = obj

            if idx < len(path)-1:
                obj = obj.as_child(key, **kwargs)
            else:
                obj = obj.as_child(key, value, default_value=default_value, **kwargs)
        # else:
        #     # obj = default_value
        #     logger.debug(f"Canot find {path[:idx]} {idx} in {self}")
        # if obj is _not_found_ or obj is None:
        #     obj = default_value

        return obj

        #    return value
        # if type_hint is None or type_hint is _not_found_:
        #     # 当 type_hint 为 None 时，，尽可能返回 raw value
        #     if isinstance(value, Entry):  # 若 value 为 entry，获得其 __value__
        #         value = value.__value__
        #     elif value is None or value is _not_found_:  # 若为 None，零七位 default_value
        #         value = default_value

        #     if isinstance(value, primary_type):  # 若 value 为 primary_type, 直接返回
        #         return value
        #     else:  # 否则转换为 Node
        #         return Node(value, default_value=default_value, **kwargs)

        # def get_child_by_key(obj: Node, path: list, type_hint=None, default_value=_not_found_, parent=None, **kwargs) -> Node:

        #     path = Path(path)

        #     if not isinstance(obj, Node):
        #         return as_node(as_entry(obj, path=path), type_hint=type_hint,
        #                        defalut_value=default_value, parent=parent, **kwargs)

        # def _validate(self, value, type_hint) -> bool:
        #     if value is _undefined_ or type_hint is _undefined_:
        #         return False
        #     else:
        #         v_orig_class = getattr(value, "__orig_class__", value.__class__)

        #         if inspect.isclass(type_hint) and inspect.isclass(v_orig_class) and issubclass(v_orig_class, type_hint):
        #             res = True
        #         elif typing.get_origin(type_hint) is not None \
        #                 and typing.get_origin(v_orig_class) is typing.get_origin(type_hint) \
        #                 and typing.get_args(v_orig_class) == typing.get_args(type_hint):
        #             res = True
        #         else:
        #             res = False
        #     return res

        # def _as_child(self, key: str, value=_not_found_,  *args, **kwargs) -> Node:
        #     raise NotImplementedError("as_child")

    def _query(self, path: Path | PathLike, *args,  **kwargs) -> typing.Any:

        path = as_path(path)

        value = path.query(self._cache, *args, **kwargs)

        if value is _not_found_ or value is None:
            value = self._entry.child(path).query(*args, **kwargs)

        return value

        missing_cache = True

        value = _not_found_

        if self._cache is not None:
            value = path.query(self._cache, default_value=_not_found_)

        if value is not _not_found_:
            missing_cache = False
        elif self._entry is not None:
            value = self._entry.child(path).get(default_value=_not_found_)

        if default_value is _not_found_:
            default_value = path.query(self._metadata.get("default_value", _not_found_), default_value=_not_found_)

        parent = self if isinstance(path[0], str) else self._parent

        n_value = type_convert(value, self.__type_hint__(path), default_value=default_value, parent=parent)

        if missing_cache:
            path.insert(self, value)

        return n_value

    def _insert(self,  *args, **kwargs) -> _T | HTree[_T] | PrimaryType:
        # try:
        #     new_path = Path().insert(self._cache, *args, quiet=False, **kwargs)
        #     res = self[new_path]
        # except KeyError:
        entry = self._entry.insert(*args, quiet=True,  **kwargs)
        res = HTree[_T](entry, parent=self)

        return res

    def _update(self,  *args, **kwargs) -> _T | HTree[_T] | PrimaryType:
        # try:
        #     Path().update(self._cache, *args, quiet=False, **kwargs)
        # except KeyError:
        self._entry.update(*args, quiet=True,  **kwargs)
        return self

    def __ior__(self, other: HTreeLike) -> HTree[_T]:
        self._update(other)
        return self

    def __iadd__(self, other: HTreeLike) -> HTree[_T]:
        self._insert(other)
        return self


Node = HTree


class Container(HTree[_T]):

    def __getitem__(self, path) -> _T | HTree[_T]: return self.get(path)


class Dict(Container[_T]):
    def __getitem__(self, path) -> _T | HTree[_T]: return self.get(path)


class List(Container[_T]):
    def __getitem__(self, path) -> _T | HTree[_T]: return self.get(path)


class AoS(List[_T]):
    """
        Array of structure
    """

    def __init__(self, *args, identifier: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._unique_id_name = id if id is not _undefined_ else "$id"
        self._cache = {}

    def combine(self, *args, default_value=None) -> _T:

        d_list = []

        default_value = deep_reduce(default_value, self._default_value)

        if default_value is not None and len(default_value) > 0:
            d_list.append(self._default_value)

        if len(args) > 0:
            d_list.extend(args)

        if self._cache is not None and len(self._cache) > 0:
            raise NotImplementedError(f"NOT IMPLEMENTET YET! {self._cache}")
            # d_list.append(as_entry(self._cache).child(slice(None)))

        if self._entry is not None:
            d_list.append(self._entry.child(slice(None)))

        type_hint = self.__type_hint__()
        return type_hint(CombineEntry({}, *d_list), parent=self._parent)

    def unique_by_id(self, id: str = "$id") -> List[_T]:
        """ 当 element 为 dict时，将具有相同 key=id的element 合并（deep_reduce)
        """
        res = {}
        for d in self.find():
            if not isinstance(d, collections.abc.Mapping):
                raise TypeError(f"{type(d)}")
            key = d.get(id, None)
            res[key] = deep_reduce(res.get(key, None), d)

        return self.__class__([*res.values()], parent=self)

    def as_child(self, key:  int | slice,  value=None, parent=_not_found_, **kwargs) -> _T:
        parent = self._parent if parent is _not_found_ or parent is None else parent
        if isinstance(key, int) and key < 0:
            key = len(self)+key

        # if not isinstance(key, int):
        #     raise NotImplementedError(f"key must be int, not {type(key)}")
        if (value is None or value is _not_found_) and isinstance(key, int):
            value = self._cache.get(key, _not_found_)

        if (value is None or value is _not_found_):
            value = self._entry.child(key)

        value = super().as_child(key, value, parent=parent, **kwargs)

        if isinstance(key, int) and value is not _not_found_:
            self._cache[key] = value

        return value


class DictProxy(HTree[_T]):

    def __getitem__(self, path: PathLike) -> typing.Any:
        return Path(path).query(self)

    def __getattr__(self, name: str) -> typing.Any:
        return super().__getitem__(name)


class ListProxy(HTree[_T]):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, path: PathLike) -> typing.Any: return self._get(Path(path))

    def __getattr__(self, name: str) -> typing.Any: return self._get(Path([name]))
    # default_value = self._default_value.get(name, None) if self._default_value is not None else None
    # return ListProxy([(getattr(obj, name, None) if not isinstance(obj, Node) else obj.get(name, default_value=default_value)) for obj in self])

    def _get(self, path: Path) -> typing.Any:
        if len(path) == 0:
            return self
        elif isinstance(path[0], dict):
            return ListProxy(self, default_value=path[0])._get(path[1:])
        else:
            default_value = path.query(self._default_value)
            return ListProxy([(Path(path).query(obj, default_value=default_value) if not isinstance(obj, HTree) else obj.get(path, default_value=default_value)) for obj in self])

    @property
    def __value__(self): return [getattr(obj, "__value__", obj) for obj in self]

    def __reduce__(self, op=None) -> typing.Any:
        res = [obj for obj in self if obj is not None]
        if len(res) == 0:
            return None
        elif len(res) == 1:
            return res[0]

        elif op is not None:
            return op(res)

        elif isinstance(res[0], np.ndarray):
            return np.sum(res)

        else:
            return res[0]
