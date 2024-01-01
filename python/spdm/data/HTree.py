from __future__ import annotations
from typing import Generator
from typing_extensions import Self
import collections.abc
import pathlib
import typing
import types
from copy import copy, deepcopy

from ..utils.logger import deprecated, logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import (
    ArrayType,
    NumericType,
    array_type,
    as_array,
    get_args,
    get_origin,
    isinstance_generic,
    numeric_type,
    serialize,
    type_convert,
)
from ..utils.uri_utils import URITuple
from .Entry import Entry, open_entry
from .Path import Path, PathLike, Query, as_path, update_tree, merge_tree


class HTreeNode:
    _metadata = {}

    @classmethod
    def _parser_args(cls, *args, _parent=None, _entry=None, **kwargs):
        if len(args) > 0 and isinstance(args[0], (collections.abc.MutableMapping, collections.abc.MutableSequence)):
            _cache = deepcopy(args[0])
            args = args[1:]
        else:
            _cache = _not_found_

        if _entry is None:
            _entry = []
        elif not isinstance(_entry, collections.abc.Sequence) or isinstance(_entry, str):
            _entry = [_entry]

        _entry = list(args) + _entry

        if isinstance(_cache, dict):
            t_entry = _cache.pop("$entry", _not_found_)
            if t_entry is _not_found_:
                pass
            elif t_entry is False:
                _entry = None
            elif not isinstance(t_entry, list):
                _entry = [t_entry] + _entry
            else:
                _entry = t_entry + _entry

        if not isinstance(_entry, list):
            raise RuntimeError(f"Can not parser _entry {_entry}")

        _entry = sum([e if isinstance(e, list) else [e] for e in _entry if e is not None and e is not _not_found_], [])

        return _cache, _entry, _parent, kwargs

    def __init__(self, *args, **kwargs) -> None:
        _cache, _entry, _parent, kwargs = HTree._parser_args(*args, **kwargs)

        self._parent = _parent

        self._entry = open_entry(_entry)

        if len(kwargs) > 0:
            self._metadata = merge_tree(self.__class__._metadata, kwargs)

        self._cache = merge_tree(deepcopy(self._metadata.get("default_value", _not_found_)), _cache)

    @property
    def empty(self) -> bool:
        """判断节点是否为空，若节点为空，返回 True，否则返回 False
        @NOTE 长度为零的 list 或 dict 不是空节点
        """
        return self._cache is None or (self._cache is _not_found_ and len(self._entry) == 0)

    def __copy__(self) -> Self:
        other = object.__new__(self.__class__)
        other._cache = deepcopy(self._cache)
        other._entry = self._entry
        other._parent = self._parent
        other._metadata = deepcopy(other._metadata)
        return other

    @classmethod
    def _do_serialize(cls, source: typing.Any, dumper: Entry | typing.Callable[..., typing.Any] | bool) -> _T:
        if source is _not_found_:
            return source if not isinstance(dumper, Entry) else dumper

        elif hasattr(source.__class__, "__serialize__"):
            return source.__serialize__(dumper)

        elif isinstance(source, dict):
            if isinstance(dumper, Entry):
                for k, v in source.items():
                    cls._do_serialize(v, dumper.child(k))
                res = dumper
            else:
                res = {k: cls._do_serialize(v, dumper) for k, v in source.items()}

        elif isinstance(source, list):
            if isinstance(dumper, Entry):
                for k, v in enumerate(source):
                    cls._do_serialize(v, dumper.child(k))
                res = dumper
            else:
                res = [cls._do_serialize(v, dumper) for v in source]

        elif isinstance(dumper, Entry):
            dumper.insert(source)
            res = dumper

        elif callable(dumper):
            res = dumper(source)

        elif dumper is True:
            res = deepcopy(source)

        else:
            res = source

        return res

    def __serialize__(self, dumper: Entry | typing.Callable[..., typing.Any] | bool = True) -> Entry | typing.Any:
        """若 dumper 为 Entry，将数据写入 Entry
           若 dumper 为 callable，将数据传入 callable 并返回结果
           若 dumper 为 True，返回数据的拷贝
           否则返回序列化后的数据

        Args:
            target (Entry, optional): 目标入口. Defaults to None.
            copier (typing.Callable[[typing.Any], typing.Any] | bool, optional): copier 拷贝器，当 target 为 None 有效。若为 True 则通过 copy 函数返回数据的拷贝. Defaults to None.

        Returns:
            typing.Any: 若 target is None，返回原始数据，否则返回 target
        """
        if self._cache is _not_found_:
            if self._entry is not None:
                return self._entry.dump(dumper)
            else:
                return self._do_serialize(self.__value__, dumper)
        else:
            return self._do_serialize(self._cache, dumper)

    @classmethod
    def __deserialize__(cls, *args, **kwargs) -> typing.Type[HTree]:
        return cls(*args, **kwargs)

    def dump(self) -> typing.Any:
        """导出数据，alias of self.__serialize__(True)"""
        return self.__serialize__(True)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.__name__}' />"

    @property
    def __name__(self) -> str:
        return self._metadata.get("name", "unnamed")

    @property
    def __value__(self) -> typing.Any:
        return self._cache

    def __array__(self) -> ArrayType:
        return as_array(self.__value__)

    @property
    def path(self) -> typing.List[str | int]:
        if self._parent is not None:
            return self._parent.path + [self.__name__]
        else:
            return [self.__name__]

    @property
    def root(self) -> HTree | None:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while p._parent is not None and getattr(p, "ids_properties", None) is None:
            p = p._parent
        return p


class HTree(HTreeNode):
    """Hierarchical Tree:

    一种层次化的数据结构，它具有以下特性：
    - 树节点也可以是列表 list，也可以是字典 dict
    - 叶节点可以是标量或数组 array_type，或其他 type_hint 类型
    - 节点可以有缓存（cache)
    - 节点可以有父节点（_parent)
    - 节点可以有元数据（metadata), 包含： 唯一标识（id), 名称（name), 单位（units), 描述（description), 标签（tags), 注释（comment)
    - 任意节点都可以通过路径访问
    - `get` 返回的类型由 `type_hint` 决定，默认为 Node

    """

    def __getitem__(self, path) -> HTree:
        return self.get(path)

    def __setitem__(self, path, value) -> None:
        self.put(path, value)

    def __delitem__(self, path) -> None:
        return self.remove(path)

    def __contains__(self, key) -> bool:
        return bool(self.fetch(key, Path.tags.exists))

    def __len__(self) -> int:
        return int(self.fetch(None, Path.tags.count))

    def __iter__(self) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        """遍历 children"""
        yield from self.children()

    def __equal__(self, other) -> bool:
        return bool(self.fetch(None, Path.tags.equal, other))

    def __update__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def put(self, path, value):
        return self.update(path, value, _idempotent=True)

    def get(self, path: PathLike, default_value=_undefined_) -> typing.Any:
        res = self.fetch(path, default_value=default_value)
        return self.__missing__(path) if res is _undefined_ else res

    def __missing__(self, path) -> typing.Any:
        raise KeyError(f"Can not find '{path}'! ")

    #############################################
    # RESTful API

    def insert(self, *args, **kwargs):
        return self.update(*args, _idempotent=False, **kwargs)

    def update(self, *args, **kwargs):
        if len(args) > 1 and isinstance(args[0], (str, int, Path)):
            path = args[0]
            args = args[1:]
        else:
            path = []

        path = as_path(path)

        if len(path) == 0:
            return self._update(None, *args, **kwargs)
        elif len(path) == 1:
            return self._update(path[0], *args, **kwargs)
        else:
            return path.update(self, *args, **kwargs)

    def remove(self, path, *args, **kwargs):
        """删除节点：
        - 将 _cahce 中 path 对应的节点设置为 None，这样访问时不会触发 _entry
        - 若 path 为 None，删除所有子节点
        """
        return as_path(path).update(self, None, *args, **kwargs)

    def fetch(self, path: PathLike, *args, **kwargs) -> typing.Any:
        path = as_path(path)

        match len(path):
            case 0:
                return self._fetch(None, *args, **kwargs)
            case 1:
                return self._fetch(path[0], *args, **kwargs)
            case _:
                return path.fetch(self, *args, **kwargs)

    def fetch_cache(self, pth, default_value: _T = _not_found_) -> _T:
        pth = as_path(pth)
        res = pth.get(self._cache, _not_found_)

        if res is _not_found_ and self._entry is not None:
            res = self._entry.get(pth, _not_found_)

        if res is _not_found_:
            res = default_value

        return res

    def children(self) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        yield from self._for_each()

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        yield from self._for_each(*args, **kwargs)

    ################################################################################
    # Private methods

    def _type_hint(self, path: PathLike = None) -> typing.Type:
        """当 key 为 None 时，获取泛型参数，若非泛型类型，返回 None，
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
                tmp = getattr(getattr(tp_hint, key, None), "type_hint", None)
                if tmp is not None:
                    tp_hint = tmp
                elif typing.get_origin(tp_hint) is None:
                    tp_hints = typing.get_type_hints(tp_hint)
                    tp_hint = tp_hints.get(key, None)
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

    def _type_convert(
        self,
        value,
        key,
        default_value=_not_found_,
        _type_hint: typing.Type = None,
        _entry: Entry | None = None,
        _parent: HTree | None = None,
        _getter: typing.Callable | None = None,
        force=False,  # 当 force == True，强制从 _entry 中取回 value
        **kwargs,
    ) -> _T:
        """ """
        # if "name" in kwargs and kwargs["name"] != key:
        #     logger.warning(f"{kwargs['name']}!={key}")

        kwargs.setdefault("name", key)
        # 整合 default_value
        s_default_value = self._metadata.get("default_value", _not_found_)
        if s_default_value is _not_found_:
            pass

        elif isinstance(key, str) and isinstance(s_default_value, dict):
            s_default_value = deepcopy(s_default_value.get(key, _not_found_))
            default_value = update_tree(s_default_value, default_value)

        elif isinstance(key, int) and isinstance(s_default_value, list) and len(s_default_value) > 0:
            s_default_value = deepcopy(s_default_value[0])
            default_value = update_tree(s_default_value, default_value)

        elif s_default_value is not _not_found_:
            # logger.debug(f"ignore {self.__class__.__name__}.{key} {s_default_value} {self._metadata}")
            pass

        if _parent is None:
            _parent = self

        if _type_hint is None:
            _type_hint = self._type_hint(key if key is not None else 0)

        if isinstance(_type_hint, typing.types.UnionType):
            tps = typing.get_args(_type_hint)
            for tp in tps:
                if isinstance_generic(value, tp):
                    _type_hint = tp
                    break
            else:
                _type_hint = tps[0]

        if isinstance_generic(value, _type_hint):
            res = value

        elif issubclass(get_origin(_type_hint), HTree) and _getter is None and force is False:
            # 当 type_hint 为 HTree 并且 _getter 为空，利用 value,_entry,default_value 创建 HTree 对象返回
            res = _type_hint(value, _entry=_entry, _parent=_parent, default_value=default_value, **kwargs)

        else:
            if value is _not_found_ and _entry is not None:
                value = _entry.get(default_value=_not_found_)

            if not isinstance_generic(value, _type_hint) and _getter is not None:
                try:
                    value = _getter(self)
                except Exception as error:
                    raise RuntimeError(f"{self.__class__} id={key}: 'getter' failed!") from error

            if value is _not_found_:
                res = default_value

            elif _type_hint is None or isinstance_generic(value, _type_hint):
                res = value

            elif issubclass(get_origin(_type_hint), HTreeNode):
                res = _type_hint(value, _parent=_parent, **kwargs)

            elif _type_hint is array_type:
                res = as_array(value)

            else:
                res = type_convert(value, _type_hint)

        if isinstance(res, HTreeNode):
            if res._parent is None:
                res._parent = _parent
            res._metadata.setdefault("name", key)

            if isinstance(key, str) and key.isidentifier():
                metadata = getattr(getattr(self.__class__, key, None), "metadata", _not_found_)
                res._metadata = merge_tree(metadata, res._metadata)

        if isinstance(key, int):
            if self._cache is _not_found_:
                self._cache = []
            elif not isinstance(self._cache, list):
                raise ValueError(self._cache)

            if key >= 0:
                pass
            else:
                key += self.__len__()

            if key >= len(self._cache):
                self._cache.extend([_not_found_] * (key - len(self._cache) + 1))

        elif isinstance(key, str):
            if self._cache is _not_found_ or self._cache is None:
                self._cache = {}

        else:
            return res

        self._cache[key] = res
        return res

    def _fetch(self, key: PathLike = None, *args, _type_hint=None, default_value=_not_found_, **kwargs) -> typing.Any:
        """获取子节点"""

        pth = Path([key])

        value = pth.fetch(self._cache, *args, default_value=_not_found_, **kwargs)

        if len(args) > 0:
            # args >0 意为包含 op ，直接返回结果，不进行类型转换
            if value is not _not_found_:
                pass
            elif self._entry is not None:
                value = self._entry.child(key).fetch(*args, default_value=default_value, **kwargs)
            else:
                value = default_value

            return value

        elif len(args) == 0:
            # 获得节点value，需要类型转换为 HTree
            _entry = self._entry.child(key) if self._entry is not None else None

            return self._type_convert(
                value, key, _entry=_entry, _type_hint=_type_hint, default_value=default_value, **kwargs
            )

    def _for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        if self._cache is _not_found_ and self._entry is not None:
            for k, v in self._entry.for_each(*args, **kwargs):
                if not isinstance(v, Entry):
                    yield k, self._type_convert(v, k)
                else:
                    yield k, self._type_convert(None, k, _entry=v)

        elif self._cache is not None:
            for k, v in Path().for_each(self._cache, *args, **kwargs):
                _entry = self._entry.child(k) if self._entry is not None else None
                v = self._type_convert(v, k, _entry=_entry)
                yield k, v

    def _update(self, key, *args, **kwargs):
        self._cache = Path._do_update(self._cache, [key], *args, **kwargs)
        return self


def as_htree(*args, **kwargs):
    if len(args) == 0 and len(kwargs) > 0:
        res = Dict(kwargs)
        kwargs = {}
    elif len(args) > 1 and len(kwargs) == 0:
        res = List(list(args))
        args = []
    elif len(args) == 0:
        res = None

    elif isinstance(args[0], HTree):
        res = args[0]
        args = args[1:]
    elif isinstance(args[0], collections.abc.MutableMapping):
        res = Dict(args[0])
        args = args[1:]
    elif isinstance(args[0], collections.abc.MutableSequence):
        res = List(args[0])
        args = args[1:]
    elif len(args) > 1:
        res = List(list(args))
        args = []

    else:
        res = HTree(*args, **kwargs)

    if len(args) + len(kwargs) > 0:
        res.update(*args, **kwargs)

    return res


Node = HTree

_T = typing.TypeVar("_T")


class Container(HTree, typing.Generic[_T]):
    """
    带有type hint的HTree，其成员类型为 _T，用于存储一组数据或对象，如列表，字典等
    """

    def __iter__(self) -> typing.Generator[typing.Tuple[int | str, _T], None, None]:
        """遍历 children"""
        yield from self.children()


class Dict(Container[_T]):
    _metadata = {"default_value": {}}

    def __copy__(self) -> Self:
        other = super().__copy__()

        if isinstance(other._cache, dict):
            for k, value in other._cache.items():
                if isinstance(value, HTreeNode):
                    value = value.__copy__()
                    value._parent = other
                    other._cache[k] = value

        return other

    def items(self) -> typing.Generator[typing.Tuple[str, _T], None, None]:
        yield from self.children()

    def keys(self) -> typing.Generator[str, None, None]:
        yield from Path.keys(self)


collections.abc.MutableMapping.register(Dict)


class List(Container[_T]):
    _metadata = {"default_value": []}

    def __copy__(self) -> Self:
        other = super().__copy__()
        if isinstance(other._cache, list):
            for k, value in enumerate(other._cache):
                if isinstance(value, HTreeNode):
                    value = value.__copy__()
                    value._parent = other
                    other._cache[k] = value

        return other

    def __iter__(self) -> Generator[_T, None, None]:
        for idx, v in self.children():
            yield v

    def append(self, other):
        self.insert(other)
        return self

    def extend(self, other):
        self.insert(other, _extend=True)
        return self

    def __iadd__(self, other: list) -> typing.Type[List[_T]]:
        return self.extend(other)


collections.abc.MutableSequence.register(List)
