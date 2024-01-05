from __future__ import annotations
from typing import Generator
from typing_extensions import Self
import collections.abc
import pathlib
import typing
import types
from copy import copy, deepcopy


from ..utils.misc import get_positional_argument_count
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

from .Entry import Entry, open_entry
from .Path import Path, PathLike, Query, as_path, update_tree, merge_tree, path_like


class HTreeNode:
    @classmethod
    def _parser_args(cls, *args, _parent=None, _entry=None, **kwargs):
        if len(args) > 0 and isinstance(args[0], (collections.abc.MutableMapping, collections.abc.MutableSequence)):
            _cache = args[0]
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

            metadata = {k[1:]: _cache.pop(k) for k in list(_cache.keys()) if k.startswith("@")}
        else:
            metadata = {}

        metadata.update(kwargs)

        if not isinstance(_entry, list):
            raise RuntimeError(f"Can not parser _entry {_entry}")

        _entry = sum([e if isinstance(e, list) else [e] for e in _entry if e is not None and e is not _not_found_], [])

        return _cache, _entry, _parent, metadata

    def __init__(self, *args, **kwargs) -> None:
        _cache, _entry, _parent, metadata = HTreeNode._parser_args(*args, **kwargs)

        self._parent = _parent

        self._entry = open_entry(_entry)

        self._metadata = deepcopy({**getattr(self.__class__, "_metadata", {}), **metadata})

        default_value = deepcopy(self._metadata.get("default_value", _not_found_))

        if _cache is _not_found_:
            self._cache = default_value
        elif isinstance(default_value, collections.abc.Mapping):
            self._cache = update_tree(deepcopy(default_value), _cache)
        else:
            self._cache = _cache

    def __copy__(self) -> Self:
        if isinstance(self._cache, dict):
            cache = {k: copy(value) for k, value in self._cache.items()}
        elif isinstance(self._cache, list):
            cache = [copy(value) for k, value in self._cache.items()]
        else:
            cache = deepcopy(self._cache)

        res = self.__class__(cache, _entry=self._entry)

        res._metadata = deepcopy(self._metadata)

        return res

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

    def __duplicate__(self, *args, _parent=_not_found_, **kwargs):
        if _parent is _not_found_:
            _parent = self._parent

        return self.__class__(
            *args,
            _parent=_parent,
            _entry=self._entry,
            **collections.ChainMap(kwargs, deepcopy(self._metadata)),
        )

    @property
    def __name__(self) -> str:
        return self._metadata.get("name", None) or self.__class__.__name__

    @property
    def __path__(self) -> typing.List[str | int]:
        if isinstance(self._parent, HTreeNode):
            return self._parent.__name__ + [self.__name__]
        else:
            return [self.__name__]

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.__name__}' />"

    @property
    def __root__(self) -> HTree | None:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while isinstance(p, HTreeNode) and getattr(p, "ids_properties", None) is None:
            p = p._parent
        return p

    @property
    def __value__(self) -> typing.Any:
        return self._cache

    def __array__(self) -> ArrayType:  # for numpy
        return as_array(self.__value__)

    def __equal__(self, other) -> bool:
        if isinstance(other, _not_found_):
            return self.__null__()
        else:
            raise NotImplementedError(f"equal operator")

    def __update__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def __null__(self) -> bool:
        """判断节点是否为空，若节点为空，返回 True，否则返回 False
        @NOTE 长度为零的 list 或 dict 不是空节点
        """
        return self._cache is None and self._entry is None

    def __empty__(self) -> bool:
        return (self._cache is _not_found_ or len(self._cache) == 0) and (self._entry is None or len(self._entry) == 0)

    def __bool__(self) -> bool:
        return self.__null__() or self.__empty__() or bool(self.__value__)

    def __getitem__(self, path) -> HTree:
        return self.get(path)

    def __setitem__(self, path, value) -> None:
        self.put(path, value)

    def __delitem__(self, path) -> None:
        return self.remove(path)

    def __contains__(self, key) -> bool:
        return bool(self.find(key, Path.tags.exists) is True)

    def __len__(self) -> int:
        if self.__empty__():
            return 0
        else:
            return int(self.find(None, Path.tags.count) or 0)

    def __iter__(self) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        """遍历 children"""
        yield from self.children()

    @staticmethod
    def _do_fetch(obj, *args, **kwargs):
        if hasattr(obj.__class__, "fetch"):
            return obj.fetch(*args, **kwargs)

        elif isinstance(obj, dict):
            return {k: HTreeNode._do_fetch(v, *args, **kwargs) for k, v in obj.items()}

        elif isinstance(obj, (list, set, tuple)):
            return obj.__class__([HTreeNode._do_fetch(v, *args, **kwargs) for v in obj])

        else:
            return obj

    def fetch(self, *args, _parent=_not_found_, **kwargs) -> Self:
        if _parent is _not_found_:
            _parent = self._parent

        cache = HTreeNode._do_fetch(self._cache, *args, _parent=None, **kwargs)

        return self.__class__(cache, _parent=_parent, _entry=self._entry, **deepcopy(self._metadata))

    # 对元素操作
    def put(self, path, value):
        return self.update(path, value, _idempotent=True)

    def get(self, path: PathLike, default_value=_undefined_) -> typing.Any:
        res = self.find(path, default_value=default_value)
        return self.__missing__(path) if res is _undefined_ else res

    def insert(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            return self
        else:
            self.__class__ = HTree
            _parent = self._parent
            self.__init__(*args, **collections.ChainMap(self._metadata, kwargs))
            if self._parent is None:
                self._parent = _parent
            return self

    def remove(self, *args, **kwargs):
        return self

    def find(self, *args, **kwargs):
        return self

    def find_cache(self, pth, default_value=_not_found_):
        return default_value

    def children(self) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        return

    def for_each(self, *args, **kwargs):
        return

    def flush(self):
        """TODO: 将 cache 内容 push 入 entry"""
        return NotImplemented


null_node = HTreeNode(_not_found_)


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
        return bool(self.find(key, Path.tags.exists) is True)

    def __len__(self) -> int:
        return int(self.find(None, Path.tags.count) or 0)

    def __iter__(self) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        """遍历 children"""
        yield from self.children()

    def __equal__(self, other) -> bool:
        return bool(self.find(None, Path.tags.equal, other))

    def __update__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def __missing__(self, path) -> typing.Any:
        raise KeyError(f"{self.__class__.__name__} can not find '{path}'! ")

    # 对元素操作
    def put(self, path, value):
        return self.update(path, value, _idempotent=True)

    def get(self, path: PathLike, default_value=_undefined_) -> typing.Any:
        value = self.find(path, default_value=_not_found_)
        if value is not _not_found_:
            return value
        elif default_value is _undefined_:
            return self.__missing__(path)
        else:
            return default_value

    def pop(self, path):
        path = as_path(path)
        value = path.get(self, _not_found_)
        if value is not _not_found_:
            path.remove(self)
        return value

    #############################################
    # RESTful API

    def insert(self, *args, **kwargs):
        return self.update(*args, _idempotent=False, **kwargs)

    def update(self, path=None, *args, **kwargs):
        if isinstance(path, dict):
            args = [path, *args]
            path = None

        path = as_path(path)

        match len(path):
            case 0:
                self._update(None, *args, **kwargs)
            case 1:
                self._update(path[0], *args, **kwargs)
            case _:
                path.update(self, *args, **kwargs)

        return self

    def remove(self, path=None, *args, **kwargs):
        path = as_path(path)

        match len(path):
            case 0:
                self._remove(None, *args, **kwargs)
            case 1:
                self._remove(path[0], *args, **kwargs)
            case _:
                path.remove(self, *args, **kwargs)

        return self

    def find(self, path=None, *args, **kwargs) -> typing.Any:
        path = as_path(path)

        match len(path):
            case 0:
                return self._find(None, *args, **kwargs)
            case 1:
                return self._find(path[0], *args, **kwargs)
            case _:
                return path.find(self, *args, **kwargs)

    def find_cache(self, path, default_value: _T = _not_found_) -> _T:
        path = as_path(path)
        res = path.get(self._cache, _not_found_)

        if res is _not_found_ and self._entry is not None:
            res = self._entry.get(path, _not_found_)

        if res is _not_found_:
            res = default_value

        return res

    def children(self) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        yield from self._for_each()

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        yield from self._for_each(*args, **kwargs)

    ################################################################################
    # Private methods

    def _get_type_hint(self, path: PathLike = None) -> typing.Type:
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
        _name,
        default_value=_not_found_,
        _type_hint: typing.Type = None,
        _entry: Entry | None = None,
        _parent: HTree | None = None,
        **kwargs,
    ) -> _T:
        if _parent is None:
            _parent = self

        if _type_hint is None:
            _type_hint = self._get_type_hint(_name if _name is not None else 0)

        if isinstance_generic(value, _type_hint):
            pass

        else:
            # 整合 default_value
            if isinstance(_name, str) and isinstance(self, collections.abc.Mapping):
                s_default_value = Path(f"default_value/{_name}").get(self._metadata, _not_found_)
            elif isinstance(self, collections.abc.Sequence):
                s_default_value = Path(f"default_value").get(self._metadata, _not_found_)
            else:
                s_default_value = _not_found_

            default_value = deepcopy(update_tree(s_default_value, default_value))

            if value is _not_found_ and _entry is None:
                value = default_value

            if (value is _not_found_ or value is _undefined_) and _entry is None:
                pass

            elif not issubclass(get_origin(_type_hint), HTree):
                if value is not _not_found_:
                    pass
                elif _entry is not None:
                    value = _entry.get(default_value=default_value)
                else:
                    value = default_value

                if value is not _not_found_:
                    value = type_convert(_type_hint, value, **kwargs)

            elif value is _not_found_ and default_value is not _undefined_:
                value = _type_hint(
                    default_value,
                    _entry=_entry,
                    _parent=_parent,
                    **kwargs,
                )

            else:
                value = _type_hint(
                    value,
                    _entry=_entry,
                    _parent=_parent,
                    default_value=default_value,
                    **kwargs,
                )

        if isinstance(value, HTreeNode):
            if value._parent is None:
                value._parent = _parent

            if isinstance(_name, str):
                value._metadata["name"] = _name

            value._metadata.update(kwargs)

        if value is not _not_found_:
            self._cache = Path._do_update(self._cache, [_name], value)

        return value

    def _update(self, key: str | int, *args, _setter=None, **kwargs):
        if _setter is not None:
            _setter(self, key, *args, **kwargs)

        elif isinstance(key, str) and key.startswith("@"):
            self._metadata = Path._do_update(self._metadata, [key[1:]], *args, **kwargs)

        elif key is None:
            self._cache = Path._do_update(self._cache, [], *args, **kwargs)

        else:
            self._cache = Path._do_update(self._cache, [key], *args, **kwargs)

        return self

    def _remove(self, key, *args, **kwargs) -> bool:
        """删除节点：
        - 将 _cahce 中 path 对应的节点设置为 None，这样访问时不会触发 _entry
        - 若 path 为 None，删除所有子节点
        """
        self._update(key, None, *args, **kwargs)

    def _find(self, _name: str | int | None, *args, default_value=_not_found_, _getter=None, **kwargs) -> typing.Any:
        """获取子节点/或属性"""

        if isinstance(_name, str) and _name.startswith("@"):
            value = Path._do_find(self._metadata, [_name[1:]], *args, default_value=_not_found_)

        elif _name is None:
            if self._cache is not _not_found_:
                if len(args) == 0:
                    value = self._cache
                else:
                    value = Path._do_find(self._cache, [], *args)
            else:
                value = _not_found_

        elif isinstance(_name, int):
            if _name < len(self._cache):
                value = self._cache[_name]
            else:
                value = _not_found_
        else:
            value = Path._do_find(self._cache, [_name], *args, default_value=_not_found_)

        if value is _not_found_ and callable(_getter):
            if get_positional_argument_count(_getter) == 2 + len(args):
                value = _getter(self, _name, *args)
            else:
                value = _getter(self)

        if len(args) > 0:
            # args >0 意为包含 op ，直接返回结果，不进行类型转换
            if value is not _not_found_:
                pass
            elif self._entry is not None:
                value = self._entry.child(_name).search(*args, default_value=default_value)
            else:
                value = default_value

        else:
            # 获得节点value，需要类型转换为 HTree
            _entry = self._entry.child(_name) if self._entry is not None else None

            value = self._type_convert(value, _name, _entry=_entry, default_value=default_value, **kwargs)

        return value

    def _for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int | str, HTreeNode], None, None]:
        if (self._cache is _not_found_ or len(self._cache) == 0) and self._entry is not None:
            for k, v in self._entry.for_each(*args, **kwargs):
                if not isinstance(v, Entry):
                    yield k, self._type_convert(v, k)
                else:
                    yield k, self._type_convert(_not_found_, k, _entry=v)

        elif self._cache is not None:
            for k, v in Path().for_each(self._cache, *args, **kwargs):
                _entry = self._entry.child(k) if self._entry is not None else None
                v = self._type_convert(v, k, _entry=_entry)
                yield k, v


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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._cache is _not_found_:
            self._cache = {}

    def get(self, *args, **kwargs) -> _T:
        return super().get(*args, **kwargs)

    def __getitem__(self, path) -> _T:
        return super().__getitem__(path)

    def items(self) -> typing.Generator[typing.Tuple[str, _T], None, None]:
        yield from self.children()

    def keys(self) -> typing.Generator[str, None, None]:
        for k, v in self.children():
            yield k

    def values(self) -> typing.Generator[_T, None, None]:
        for k, v in self.children():
            yield v


collections.abc.MutableMapping.register(Dict)


class List(Container[_T]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._cache is _not_found_:
            self._cache = []

    def __iter__(self) -> Generator[_T, None, None]:
        for idx, v in self.children():
            yield v

    def append(self, other):
        self.update(None, other, _idempotent=False, _extend=False)
        return self

    def extend(self, other):
        self.update(None, other, _idempotent=False, _extend=True)
        return self

    def __iadd__(self, other: list) -> typing.Type[List[_T]]:
        return self.extend(other)


collections.abc.MutableSequence.register(List)
