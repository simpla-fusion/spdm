from __future__ import annotations

import ast
import collections.abc
import pprint
import re
import typing
from copy import copy, deepcopy
from enum import Flag, auto

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import array_type
from ..utils.tree_utils import merge_tree_recursive
_T = typing.TypeVar("_T")


class OpTags(Flag):
    # traversal operation 操作

    next = auto()
    parent = auto()
    current = auto()
    root = auto()

    # crud operation
    insert = auto()
    append = auto()
    extend = auto()
    update = auto()
    remove = auto()
    deep_update = auto()
    setdefault = auto()
    reduce = auto()
    merge = auto()

    # for sequence
    sort = auto()

    # predicate 谓词
    check = auto()
    count = auto()

    # boolean
    equal = auto()
    le = auto()
    ge = auto()
    less = auto()
    greater = auto()


PathLike = int | str | slice | typing.Dict | typing.List | OpTags | None

path_like = (int, str, slice, list, None, tuple, set, dict, OpTags)


class Path(typing.List[PathLike]):
    """
    Path用于描述数据的路径, 在 JSON DOM Tree (Entry) 中定位Element, 其语法是 JSONPath 和 XPath的变体，并扩展谓词（predicate）语法/查询选择器。

    JSON DOM Tree : 半结构化树状数据，子树节点具有 list或dict类型，叶节点为 list和dict 之外的primary数据类型，包括 int，float,string 和 ndarray。

    基本原则是用python 原生数据类型（例如，list, dict,set,tuple）等

    DELIMITER=`/` or `.`

    | Python 算符                   | 字符形式                   | 描述
    | ----                          |---                        | ---
    | N/A                           | `$`                       | 根对象 （ TODO：Not Implemented ）
    | None                          | `@`                       | 空选择符，当前对象。当以Path以None为最后一个item时，表示所指元素为leaf节点。
    | `__truediv__`,`__getattr___`  | DELIMITER (`/` or `.`)    | 子元素选择符, DELIMITER 可选
    | `__getitem__`                 | `[index|slice|selector]`| 数组元素选择符，index为整数,slice，或selector选择器（predicate谓词）

    predicate: 谓词, 过滤表达式，用于过滤数组元素.
    | `set`                         | `[{a,b,1}]`               | 返回dict, named并集运算符，用于组合多个子元素选择器，并将element作为返回的key， {'a':@[a], 'b':@['b'], 1:@[1] }
    | `list`                        | `["a",b,1]`               | 返回list, 并集运算符，用于组合多个子元素选择器，[@[a], @['b'], @[1]]
    | `slice`                       | `[start:end:step]`，      | 数组切片运算符, 当前元素为 ndarray 时返回数组切片 @[<slice>]，当前元素为 dict,list 以slice选取返回 list （generator），
    | `slice(None) `                | `*`                       | 通配符，匹配任意字段或数组元素，代表所有子节点（children）
    |                               | `..`                      | 递归下降运算符 (Not Implemented)
    | `dict` `{$eq:4, }`            | `[?(expression)]`         | 谓词（predicate）或过滤表达式，用于过滤数组元素.
    |                               | `==、!=、<、<=、>、>=`     | 比较运算符

    examples：
    | Path                              | Description
    | ----                              | ---
    | `a/b/c`                           | 选择a节点的b节点的c节点
    | `a/b/c/1`                         | 选择a节点的b节点的c节点的第二个元素
    | `a/b/c[1:3]`                      | 选择a节点的b节点的c节点的第二个和第三个元素
    | `a/b/c[1:3:2]`                    | 选择a节点的b节点的c节点的第二个和第三个元素
    | `a/b/c[1:3:-1]`                   | 选择a节点的b节点的c节点的第三个和第二个元素
    | `a/b/c[d,e,f]`                    |
    | `a/b/c[{d,e,f}]                   |
    | `a/b/c[{value:{$le:10}}]/value    |
    | `a/b/c.$next/                     |
    主要的方法：
    find
    """
    tags = OpTags
    DELIMITER = '/'

    def __init__(self, path=None, delimiter=None, **kwargs):
        if delimiter is None:
            delimiter = Path.DELIMITER
        path = Path._parser(path, delimiter=delimiter)

        super().__init__(path, **kwargs)

        self._delimiter = delimiter

    def __repr__(self): return Path._to_str(self)

    def __str__(self): return Path._to_str(self)

    def __hash__(self) -> int: return self.__str__().__hash__()

    def __copy__(self) -> Path: return self.__class__(self[:])

    def as_url(self) -> str: return Path._to_str(self, delimiter=self._delimiter)

    @property
    def is_regular(self) -> bool:
        return next((i for i, v in enumerate(self) if not isinstance(v, Path._PRIMARY_INDEX_TYPE_)), None) is None

    @property
    def is_generator(self) -> bool: return any([isinstance(v, (slice, dict)) for v in self])

    @property
    def parent(self) -> Path:
        # if self.is_root:
        #     raise RuntimeError("Root node hasn't parents")
        other = copy(self)
        other.pop()
        return other

    @property
    def children(self) -> Path:
        # if self.is_leaf:
        #     raise RuntimeError("Leaf node hasn't child!")
        other = copy(self)
        other.append(slice(None))
        return other

    @property
    def slibings(self): return self.parent.children

    @property
    def next(self) -> Path:
        other = copy(self)
        other.append(Path.tags.next)
        return other

    def resolve(self) -> Path:
        """
            - reduce  path to regular path, 执行 Path.tags.parent ,current, root，

        """
        # 查找第二个以后的元素，若为 Path.tags.parent ,current, root 删除相应的单元

        idx = 1

        while idx > 0 and idx < len(self):
            p = super().__getitem__(idx)
            if not isinstance(p, Path.tags):
                idx += 1
                continue
            elif p is Path.tags.parent:
                del self[idx-1:idx+1]
                idx -= 2

            elif p is Path.tags.current:
                self.pop(idx)
            elif p is Path.tags.root:
                del self[:idx]
                idx = 0
            else:
                idx += 1
        return self

    def append(self, d) -> Path:
        if isinstance(d, str):
            d = Path._from_str(d, delimiter=self._delimiter)
        Path._unroll(d, self)
        return self

    def extend(self, *args, force=False) -> Path: return Path._unroll(args, self)

    def __truediv__(self, p) -> Path: return copy(self).append(p)

    def __add__(self, p) -> Path: return copy(self).append(p)

    def __iadd__(self, p) -> Path: return self.append(p)

    def __eq__(self, other) -> bool:
        if isinstance(other, list):
            return super().__eq__(other)
        elif isinstance(other, Path):
            return super().__eq__(other[:])
        else:
            return False

    ###########################################################
    # API: CRUD  operation

    def find(self, target: typing.Any, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        yield from Path._find(target, self[:], *args, **kwargs)

    def traversal(self) -> typing.Generator[typing.List[typing.Any], None, None]:
        yield from Path._traversal_path(self[:])

    def query(self, target: typing.Any, *args, **kwargs) -> typing.Any:
        return Path._query(target, self[:], *args, **kwargs)

    def insert(self, target: typing.Any, *args, create_if_not_exists=True, **kwargs) -> typing.Any:
        if create_if_not_exists is False:
            pass
        elif isinstance(self[-1], str):
            create_if_not_exists = {}
        elif not isinstance(self[-1], Path.tags):
            create_if_not_exists = []
        else:
            create_if_not_exists = None
        return Path._query(target, self[:-1], Path.tags.insert, self[-1], *args, create_if_not_exists=create_if_not_exists, **kwargs)

    def update(self, target: typing.Any, *args, create_if_not_exists=False,  **kwargs) -> int:
        if create_if_not_exists is False:
            pass
        elif isinstance(self[-1], str):
            create_if_not_exists = {}
        elif not isinstance(self[-1], Path.tags):
            create_if_not_exists = []
        else:
            create_if_not_exists = None
        return Path._query(target, self[:-1], Path.tags.update, self[-1], *args,  create_if_not_exists=create_if_not_exists, **kwargs)

    def remove(self, target: typing.Any, *args, **kwargs) -> int:
        return Path._query(target, self[:-1], Path.tags.remove, self[-1], *args, **kwargs)

    # End API
    ###########################################################

    _PRIMARY_INDEX_TYPE_ = (int, float)

    @staticmethod
    def reduce(path: list) -> list:
        if len(path) < 2:
            return path
        elif isinstance(path[0], set) and path[1] in path[0]:
            return Path.reduce(path[1:])
        elif isinstance(path[0], slice) and isinstance(path[1], int):
            start = path[0].start if path[0].start is not None else 0
            step = path[0].step if path[0].step is not None else 1
            stop = start+step*path[1]
            if path[0].stop is not None and stop > path[0].stop:
                raise IndexError(f"index {stop} is out of range")
            return [stop, *Path.reduce(path[2:])]
        else:
            return path

    @staticmethod
    def normalize(p: typing.Any, raw=False) -> typing.Any:
        if p is None:
            res = []
        elif isinstance(p, Path):
            res = p[:]
        elif isinstance(p, str):
            res = Path._from_str(p)
        elif isinstance(p, (int, slice)):
            res = p
        elif isinstance(p, list):
            res = sum((([v] if not isinstance(v, list) else v) for v in map(Path.normalize, p)), list())
        elif isinstance(p, tuple):
            if len(p) == 1:
                res = Path.normalize(p[0])
            else:
                res = tuple(map(Path.normalize, p))
        elif isinstance(p, collections.abc.Set):
            res = set(map(Path.normalize, p))
        elif isinstance(p, collections.abc.Mapping):
            res = {Path.normalize(k): Path.normalize(v, raw=True) for k, v in p.items()}
        else:
            res = p
            # raise TypeError(f"Path.normalize() only support str or Path, not {type(p)}")

        # if not raw and not isinstance(res, list):
        #     res = [res]

        return res

    # example:
    # a/b_c6/c[{value:{$le:10}}][value]/D/[1，2/3，4，5]/6/7.9.8
    PATH_REGEX = re.compile(r"(?P<key>[^\[\]\/\,\.]+)|(\[(?P<selector>[^\[\]]+)\])")

    # 正则表达式解析，匹配一段被 {} 包裹的字符串
    PATH_REGEX_DICT = re.compile(r"\{(?P<selector>[^\{\}]+)\}")

    @staticmethod
    def _to_str(p: typing.Any, delimiter=None) -> str:
        if delimiter is None:
            delimiter = Path.DELIMITER

        if isinstance(p, list):
            return delimiter.join(map(Path._to_str, p))
        elif isinstance(p, str):
            return p
        elif isinstance(p, slice):
            if p.start is None and p.stop is None and p.step is None:
                return "*"
            else:
                return f"{p.start}:{p.stop}:{p.step}"
        elif isinstance(p, int):
            return str(p)
        elif isinstance(p, collections.abc.Mapping):
            m_str = ','.join([f"{k}:{Path._to_str(v)}" for k, v in p.items()])
            return f"?{{{m_str}}}"
        elif isinstance(p, tuple):
            m_str = ','.join(map(Path._to_str, p))
            return f"({m_str})"
        elif isinstance(p, set):
            m_str = ','.join(map(Path._to_str, p))
            return f"{{{m_str}}}"
        elif p is None:
            return ""
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    @staticmethod
    def _from_str_one(s: str | list, delimiter=None) -> list | dict | str | int | slice | Path.tags:
        if isinstance(s, str):
            s = s.strip(" ")

        if not isinstance(s, str):
            item = s
        elif s.startswith(("[", "(", "{")) and s.endswith(("}", ")", "]")):
            tmp = ast.literal_eval(s)
            if isinstance(tmp, dict):
                item = {Path._from_str_one(k, delimiter=delimiter): d for k, d in tmp.items()}
            elif isinstance(tmp, set):
                item = set([Path._from_str_one(k, delimiter=delimiter) for k in tmp])
            elif isinstance(tmp, tuple):
                item = tuple([Path._from_str_one(k, delimiter=delimiter) for k in tmp])
            elif isinstance(tmp, list):
                item = [Path._from_str_one(k, delimiter=delimiter) for k in tmp]

        elif s.startswith("(") and s.endswith(")"):
            tmp: dict = ast.literal_eval(s)
            item = {Path._from_str_one(k, delimiter=delimiter): d for k, d in tmp.items()}
        elif ":" in s:
            tmp = s.split(":")
            if len(tmp) == 2:
                item = slice(int(tmp[0]), int(tmp[1]))
            elif len(tmp) == 3:
                item = slice(int(tmp[0]), int(tmp[1]), int(tmp[2]))
            else:
                raise ValueError(f"Invalid slice {s}")
        elif s == "*":
            item = slice(None)
        elif s == "..":
            item = Path.tags.parent
        elif s == ".":
            item = Path.tags.current
        elif s.isnumeric():
            item = int(s)
        elif s.startswith("$"):
            try:
                item = Path.tags[s[1:]]
            except Exception:
                item = s
        else:
            item = s

        return item

    @staticmethod
    def _from_str(path: str | list, delimiter=None) -> list:
        """ Parse the path string to list  """

        if delimiter is None:
            delimiter = Path.DELIMITER

        if isinstance(path, str):
            path = path.split(delimiter)
        elif not isinstance(path, list):
            path = [path]

        if path[0] == '':
            path[0] = Path.tags.root

        return [Path._from_str_one(v) for v in path]

    @staticmethod
    def _unroll(source: PathLike | list, target: typing.List[PathLike]) -> typing.List[PathLike]:
        """ Parse the  to list """
        if source is None:
            return target
        elif isinstance(source, str):
            source = [source]
        elif not isinstance(source, collections.abc.Sequence):
            source = [source]

        for p in source:
            if isinstance(p, str):
                list.append(target, p)
            elif isinstance(p, list):
                Path._unroll(p, target)
            elif p is Path.tags.parent:
                if len(target) > 0 and target[-1] is not Path.tags.parent:
                    target.pop()
                else:
                    list.append(target, p)
            elif p is Path.tags.root:
                target.clear()
                list.append(target, Path.tags.root)
            else:
                list.append(target, p)

        return target

    @staticmethod
    def _parser(path: PathLike, delimiter=None) -> list:
        if path is None:
            path = []
        elif isinstance(path, str):
            path = Path._from_str(path, delimiter=delimiter)
        elif isinstance(path, list):
            path = sum([(Path._from_str(p, delimiter=delimiter) if isinstance(p, str) else [p]) for p in path], [])
        else:
            path = [path]
        return Path._unroll(path, [])

    @staticmethod
    def _reduce(path: list) -> list:
        pos = 1
        while pos > 0 and pos < len(path):
            p = path[pos]
            if p is Path.tags.parent and pos > 0 and path[pos-1] is not Path.tags.parent:
                del path[pos-1:pos+1]
                pos -= 1
            elif p is Path.tags.root:
                del path[:pos]
                pos = 0
            else:
                pos += 1

        return path

    MAX_SLICE_STOP = 1024

    @staticmethod
    def _traversal_path(path: typing.List[typing.Any], prefix: typing.List[typing.Any] = []) -> typing.Generator[typing.List[type.Any], None, None]:
        """
        traversal all possible path
        """
        if len(path) == 0:
            yield prefix
            return
        try:
            pos = next(idx for idx, item in enumerate(path) if not isinstance(item, (int, str)))
        except StopIteration:
            yield path
            return
        prefix = prefix+path[:pos]
        suffix = path[pos+1:]
        item = path[pos]

        if isinstance(item, (tuple, set)):
            for k in item:
                yield from Path._traversal_path(suffix, prefix+[k])
        elif isinstance(item, collections.abc.Mapping):
            yield from Path._traversal_path(suffix, prefix+[item])
        elif isinstance(item, slice):
            start = item.start if item.start is not None else 0
            step = item.step if item.step is not None else 1
            stop = item.stop if item.stop is not None else Path.MAX_SLICE_STOP

            for k in range(start, stop, step):
                yield from Path._traversal_path(suffix, prefix+[k])

            if stop == Path.MAX_SLICE_STOP:
                logger.warning(f"MAX_SLICE_STOP, slce.stop is not defined! ")

    @staticmethod
    def _find(target: typing.Any, path: typing.List[typing.Any], *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        target, pos = Path._traversal(target, path)

        def concate_path(p, others: list):
            if not isinstance(p, list):
                return [p]+others
            else:
                return p+others
        if target is None:
            return
        elif len(path) == pos:
            yield target  # target is the last node
        elif hasattr(target.__class__, "_as_child"):
            yield from Path._find(target._as_child(path[pos]), path[pos+1:], *args, **kwargs)
        elif hasattr(target, "__entry__") and not hasattr(target.__class__, "_as_child"):
            yield target.__entry__.child(path[pos:]).find(*args, **kwargs)
        elif isinstance(path[pos], str):
            yield from Path._find(target.get(path[pos]), path[pos+1:], **kwargs)

            # raise TypeError(f"{path[pos]}")
        elif isinstance(path[pos], set):
            yield from ((k, Path._find(target, concate_path(k, path[pos+1:]),  *args, **kwargs)) for k in path[pos])
        elif isinstance(path[pos], tuple):
            yield from (Path._find(target, concate_path(k, path[pos+1:]),  *args, **kwargs) for k in path[pos])
        elif isinstance(path[pos], slice):
            if isinstance(target, (array_type)):
                yield from Path._find(target[path[pos]], path[pos+1:],  *args, **kwargs)
            elif isinstance(target, (collections.abc.Sequence)) and not isinstance(target, str):
                for item in target[path[pos]]:
                    yield from Path._find(item, path[pos+1:],  *args, **kwargs)
            elif isinstance(target, (collections.abc.Mapping)):
                target_ = {k: v for k, v in target.items() if k is not None}
                start = path[pos].start if path[pos].start is not None else 0
                stop = path[pos].stop if path[pos].start is not None else len(target_)
                step = path[pos].step if path[pos].step is not None else 1

                for k in range(start, stop, step):
                    yield from Path._find(target_[k], path[pos+1:],  *args, **kwargs)
            # elif "default_value" in kwargs:
            #     yield kwargs["default_value"]
            else:
                raise TypeError(f"Cannot slice target={(target)} path=[{path[:pos]} ^, {path[pos:]}]")
        elif isinstance(path[pos], collections.abc.Mapping):
            only_first = kwargs.get("only_first", False) or path[pos].get("@only_first", True)
            if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
                for element in target:
                    if Path._match(element, path[pos]):
                        yield from Path._find(element,  path[pos+1:],  *args, **kwargs)
                        if only_first:
                            break
            # elif "default_value" in kwargs:
            #     yield [kwargs["default_value"]]
            else:
                raise TypeError(f"Cannot search {type(target)}")
        elif "default_value" in kwargs:
            yield kwargs["default_value"]
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {path[pos]}")

    @staticmethod
    def _find_all(target: typing.Any, path: typing.List[typing.Any], *args, **kwargs):
        return Path._expand(Path._find(target, path, *args, **kwargs))

    @staticmethod
    def _query_exec(target: typing.Any,  *args, **kwargs) -> typing.Any:
        if hasattr(target, "__entry__"):
            return target.__entry__.query(*args, **kwargs)
        elif len(args) == 0:
            return target
        else:
            op = args[0]
            args = args[1:]

        if isinstance(op, Path.tags):

            if op is Path.tags.parent:
                return getattr(target, "_parent", _not_found_)
            elif op is Path.tags.root:
                parent = getattr(target, "_parent", _not_found_)
                while parent is not _not_found_:
                    target = parent
                    parent = getattr(target, "_parent", _not_found_)
                return target

            else:
                _op = getattr(Path, f"_op_{op.name}", None)

                if not callable(_op):
                    raise RuntimeError(f"Invalid operator {op}!")

                return _op(target, *args, **kwargs)

        # elif len(args) > 0:
        #     if isinstance(args[0], Path.tags):
        #         return Path._query_exec(target, args[0], op, *args[1:], **kwargs)
        #     else:
        #         raise NotImplementedError(f"Not implemented query! '{op}' {args}")

        elif isinstance(op, str):
            tmp = getattr(target, op, _not_found_)
            if tmp is _not_found_ and hasattr(target, "get"):
                tmp = target.get(op, _not_found_)
            if tmp is _not_found_ and hasattr(target, "__getitem__"):
                try:
                    tmp = target[op]
                except Exception:
                    tmp = _not_found_

            return tmp

        elif isinstance(target, array_type) and isinstance(op, (int, slice, tuple)):
            return target[op]

        elif isinstance(target, collections.abc.Sequence) and isinstance(op, (int, slice)):
            return target[op]

        elif isinstance(op, set):
            return {Path._query_exec(target, p, *args, **kwargs) for p in op}

        elif isinstance(op, dict):
            return all([Path._query_exec(target, *kv, *args, **kwargs) for kv in op.items()])

        else:
            raise NotImplementedError(f"Not implemented query! '{op}'")

    @staticmethod
    def _query(target: typing.Any, path: typing.List[typing.Any],  *args, default_value=_not_found_, create_if_not_exists: typing.Any = False, **kwargs) -> typing.Any:
        if path is None:
            path = []
        length = len(path)
        data = target
        idx = 0
        while idx >= 0 and idx < length:
            p = path[idx]

            if target is _not_found_ or target is None:
                break
            #     raise RuntimeError(f"Invalid path {path[:idx+1]} ")

            if hasattr(target, "__entry__"):
                target = target.__entry__.child(path[idx:])
                idx = length-1
            else:
                try:
                    tmp = Path._query_exec(target, p)
                except Exception as error:
                    raise RuntimeError(f"Error when execute {path[:idx+1]} on {target}, ({args}, {kwargs})") from error

                if tmp is _not_found_ and create_if_not_exists is not False:
                    if idx < length-1:
                        if isinstance(path[idx+1], str):
                            tmp = {}
                        else:
                            tmp = []
                    else:
                        tmp = create_if_not_exists

                    Path._query_exec(target, Path.tags.insert, p, tmp)
                else:
                    target = tmp
                    idx += 1

        if idx < length-1:
            # raise RuntimeError(f"Can not find {path[:idx+1]} from {type(data)} {target}! ")
            target = _not_found_
        else:
            target = Path._query_exec(target, *args,  **kwargs)

        if target is _not_found_:
            target = default_value

        return target

        # target, pos = Path._traversal(target, path)

        # if target is None or target is _not_found_:
        #     return kwargs.get("default_value", _not_found_)

        # elif pos >= len(path):  # target is the last node
        #     return target

        # elif hasattr(target, "__entry__"):  # 当 target 为 Entry 时转为 Entry 的查询
        #     return target.__entry__.child(path[pos:]).query(*args, **kwargs)

        # elif isinstance(path[pos], (dict, set, tuple, Path.tags)):  # 执行 query
        #     target = Path._query_exec(target, path[pos], **kwargs)

        #     if pos < len(path)-1:
        #         target = Path._query(target, path[pos+1:], *args, **kwargs)

        #     return target

        # else:
        #     return kwargs.get("default_value", _not_found_)
        #     # raise NotImplementedError(f"Not implemented yet! {path[pos]} {path} {target}")

    @staticmethod
    def _op_find(target, k, default_value=_undefined_):
        obj, key = Entry._eval_path(self, k, force=False, lazy=False)
        if obj is _not_found_:
            obj = default_value
        elif isinstance(key, (int, str, slice)):
            obj = obj[key]
        elif isinstance(key, list):
            obj = [obj[idx] for idx in key]
        else:
            raise TypeError(type(key))
        return obj
        # if isinstance(self, collections.abc.Mapping):
        # elif isinstance(self, collections.abc.Sequence):
        # else:
        #     raise NotImplementedError(type(self))

    @staticmethod
    def _op_by_filter(target, pred, op,  *args, on_fail: typing.Callable = _undefined_):
        if not isinstance(target, collections.abc.Sequence):
            raise TypeError(type(target))

        if isinstance(pred, collections.abc.Mapping):
            def pred(val, _cond=pred):
                if not isinstance(val, collections.abc.Mapping):
                    return False
                else:
                    return all([val.get(k, _not_found_) == v for k, v in _cond.items()])

        res = [op(target, idx, *args)
               for idx, val in enumerate(target) if pred(val)]

        if len(res) == 1:
            res = res[0]
        elif len(res) == 0 and on_fail is not _undefined_:
            res = on_fail(target)
        return res

    @staticmethod
    def _op_assign(target, path, v):
        target, key = Entry._eval_path(
            target,  Entry.normalize_path(path), force=True, lazy=False)
        if not isinstance(key, (int, str, slice)):
            raise KeyError(path)
        elif not isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
            raise TypeError(type(target))
        target[key] = v
        return v

    @staticmethod
    def _op_insert(target, k, v):
        target[k] = v
        return target[k]

    @staticmethod
    def _op_append(target, v): return target.append(v)

    @staticmethod
    def _op_remove(target, k):
        if isinstance(k, (str, int, slice)):
            try:
                del target[k]
            except Exception as error:
                success = False
            else:
                success = True
        else:
            raise NotImplementedError(f"{k}")
        return success

    @staticmethod
    def _op_update(target, value, *args, **kwargs): return merge_tree_recursive(target, value, *args, **kwargs)

    @staticmethod
    def _op_check(pred=None, *args) -> bool:

        if isinstance(pred, Entry.op_tag):
            return Entry._ops[pred](target, *args)
        elif isinstance(pred, collections.abc.Mapping):
            return all([Entry._op_check(Entry._eval_path(target, Entry.normalize_path(k), _not_found_), v) for k, v in pred.items()])
        else:
            return target == pred

    @staticmethod
    def _op_exist(target, key=_not_found_, *args, **kwargs) -> bool:
        if key is _not_found_:
            return target is not _not_found_
        elif hasattr(target, "__contains__"):
            return target.__contains__(key)
        else:
            raise TypeError(type(target))

    @staticmethod
    def _op_equal(target, other, *args, **kwargs):
        return target == other

    @staticmethod
    def _op_count(target, *args, **kwargs) -> int:
        if target is _not_found_:
            return 0
        elif not isinstance(target, (collections.abc.Sequence, collections.abc.Mapping)) or isinstance(target, str):
            return 1
        else:
            return len(target)


    # fmt: off
    _op_neg         =np.negative     
    _op_add         =np.add          
    _op_sub         =np.subtract     
    _op_mul         =np.multiply     
    _op_matmul      =np.matmul       
    _op_truediv     =np.true_divide  
    _op_pow         =np.power        
    _op_eq          =np.equal        
    _op_ne          =np.not_equal    
    _op_lt          =np.less         
    _op_le          =np.less_equal   
    _op_gt          =np.greater      
    _op_ge          =np.greater_equal
    _op_radd        =np.add          
    _op_rsub        =np.subtract     
    _op_rmul        =np.multiply     
    _op_rmatmul     =np.matmul       
    _op_rtruediv    =np.divide       
    _op_rpow        =np.power        
    _op_abs         =np.abs          
    _op_pos         =np.positive     
    _op_invert      =np.invert       
    _op_and         =np.bitwise_and  
    _op_or          =np.bitwise_or   
    _op_xor         =np.bitwise_xor  
    _op_rand        =np.bitwise_and  
    _op_ror         =np.bitwise_or   
    _op_rxor        =np.bitwise_xor  
    _op_rshift      =np.right_shift  
    _op_lshift      =np.left_shift   
    _op_rrshift     =np.right_shift  
    _op_rlshift     =np.left_shift   
    _op_mod         =np.mod          
    _op_rmod        =np.mod          
    _op_floordiv    =np.floor_divide 
    _op_rfloordiv_  =np.floor_divide 
    _op_trunc       =np.trunc        
    _op_round       =np.round        
    _op_floor       =np.floor        
    _op_ceil        =np.ceil         
    # fmt: on


class old_path:
    @staticmethod
    def _expand(target: typing.Any):
        if isinstance(target, collections.abc.Generator):
            res = [Path._expand(v) for v in target]
            if len(res) > 1 and isinstance(res[0], tuple) and len(res[0]) == 2:
                res = dict(*res)
            elif len(res) == 1:
                res = res[0]
        else:
            res = target

        return res

    @staticmethod
    def _match(target: typing.Any, predicate: typing.Mapping[str, typing.Any]) -> bool:
        """

        """
        if not isinstance(predicate, collections.abc.Mapping):
            predicate = {predicate: None}

        def do_match(op, value, expected):
            res = False
            if isinstance(op, str) and isinstance(value, collections.abc.Mapping):
                return value.get(op, _not_found_) == expected
            else:
                raise NotImplementedError(f" {op}:{expected}")

        return all([do_match(op, target, args) for op, args in predicate.items()])

    @staticmethod
    def _traversal(target: typing.Any, path: typing.List[typing.Any]) -> typing.Tuple[typing.Any, int]:
        """
        Traversal the target with the path, return the last regular target and the position the first non-regular path.
        :param target: the target to traversal
        :param path: the path to traversal

        """

        pos = -1

        for idx, p in enumerate(path):
            tmp = _not_found_

            if target is None or target is _not_found_:
                break

            elif hasattr(target, "__entry__"):
                break

            elif isinstance(p, str):
                tmp = getattr(target, p, _not_found_)
                if tmp is _not_found_ and isinstance(target, collections.abc.Mapping):
                    tmp = target.get(p, _not_found_)

            elif isinstance(target, array_type) and isinstance(p, (int, slice, tuple)):
                tmp = target[p]

            elif isinstance(target, collections.abc.Sequence) and isinstance(p, int):
                tmp = target[p]

            if tmp is _not_found_:
                break

            target = tmp  # type:ignore
            pos = idx

            # if p is _not_found_:
            #     raise TypeError(f"Cannot get '{path[:idx+1]}' in {pprint.pformat(target)}")

        return target, pos+1

    @staticmethod
    def _insert(target: typing.Any, path: typing.List[typing.Any], value: typing.Any, *args, parents=True, **kwargs) -> int:
        target, pos = Path._traversal(target, path[: -1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).insert(value, *args, parents=parents, **kwargs)
        elif len(path) == 0:
            return Path._update(target, value, *args, **kwargs)
        elif pos < len(path)-1 and not isinstance(path[pos], (int, str)):
            return sum(Path._insert(d, path[pos+1:], value, *args, **kwargs) for d in Path._find(target, [path[pos]], **kwargs))
        elif not parents:
            raise IndexError(f"Can't insert {value} to {target} by {path[:pos]}!")
        else:
            for p in path[pos: -1]:
                target = target.setdefault(p, {})
            target[path[-1]] = value
            return 1

    @staticmethod
    def _remove(target: typing.Any, path: typing.List[typing.Any],  *args, **kwargs) -> int:
        """
        Remove target by path.
        """

        target, pos = Path._traversal(target, path[: -1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).delete(*args, **kwargs)
        elif len(path) == 0:
            target.clear()
            return 1
        elif pos < len(path)-1 and not isinstance(path[pos], (int, str)):
            return sum(Path._remove(d, path[pos+1:], *args, **kwargs) for d in Path._find(target, [path[pos]], **kwargs))
        elif pos < len(path)-1:
            return 0
        else:
            del target[path[-1]]
            return 1

    @staticmethod
    def _update_or_replace(target: typing.Any, actions: collections.abc.Mapping,
                           force=False, replace=True, **kwargs) -> typing.Any:
        if not isinstance(actions, dict):
            return actions

        for op, args in actions.items():
            if isinstance(op, str) and op.startswith("$"):
                op = Path.tags[op[1:]]

            if isinstance(op, str):
                if target is None:
                    target = {}
                Path._update(target, [op], args, force=force, **kwargs)
            elif op in (Path.tags.append,  Path.tags.extend):
                new_obj = Path._update_or_replace(None, args, force=True, replace=True)
                if op is Path.tags.append:
                    new_obj = [new_obj]

                if isinstance(target, list):
                    target += new_obj
                elif replace:
                    if target is not None:
                        target = [target] + new_obj
                    else:
                        target = new_obj
                else:
                    raise IndexError(f"Can't append {new_obj} to {target}!")
            else:
                raise NotImplementedError(f"Not implemented yet!{op}")

        return target

    @staticmethod
    def _update(target: typing.Any, path: typing.List[typing.Any], value:  typing.Any = None, *args,  force=False, **kwargs) -> int:
        """
        Update target by path with value.
        """
        target, pos = Path._traversal(target, path[:-1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).update(value,  *args, overwrite=force, **kwargs)
        elif len(path) == 0:
            Path._update_or_replace(target, value, force=force, replace=False, **kwargs)
            return 1
        elif not isinstance(path[pos], (int, str)):
            return sum(Path._update(d, path[pos+1:], value, *args,  force=force, **kwargs)
                       for d in Path._find(target, [path[pos]], **kwargs))
        elif not isinstance(value, collections.abc.Mapping):
            return Path._insert(target, path[pos:], value, *args, force=force, **kwargs)
        elif pos < len(path)-1:
            return Path._insert(target, path[pos:], Path._update_or_replace(None, value), force=force, **kwargs)
        else:  # pos == len(path)-1
            n_target, n_pos = Path._traversal(target, path[pos:])
            if n_pos == 0:
                return Path._update(n_target, path[n_pos:], Path._update_or_replace(None, value), force=force, **kwargs)

            else:
                return Path._insert(target, path[pos:], Path._update_or_replace(n_target, value), force=force, **kwargs)

    @staticmethod
    def do_update(self, key, value):
        # if not isinstance(value, collections.abc.Mapping)\
        #         or not any(map(lambda k: isinstance(k, Entry.op_tag), value.keys())):
        #     try:
        #         self[key] = value
        #     except (KeyError, IndexError) as error:
        #         logger.exception(error)
        #         raise KeyError(key)
        # else:
        #     for op, v in value.items():
        #         if not isinstance(op, Entry.op_tag):
        #             logger.warning(f"Ignore illegal op {op}!")
        #         Entry._eval_op(op, self, key, v)

        return self

    @staticmethod
    def _eval_path(self, path: list, force=False) -> Tuple[Any, Any]:
        """
            Return: 返回path中最后一个key,这个key所属于的Tree node

            if force is Ture then:
                当中间node不存在时,根据key的类型创建之,
                    key is str => node is dict
                    key is int => node is list
            else:
                当中间node不存在时,返回最后一个有效node和后续path

            if key is _next_:
                append _undefined_ to current node and set key =len(curren node)
            if isisntance(key,dict)
                filter(current node, predication=key)
                key is the index of first node
        """

        if isinstance(self, Entry):
            raise NotImplementedError("Entry._eval_path do not accept Entry")
        elif self is _undefined_ or self is None or self is _not_found_:
            return _not_found_, path

        elif path is _undefined_:
            return self, None
        elif not isinstance(path, list):
            path = [path]

        last_index = len(path)-1

        val = self
        key = None
        for idx, key in enumerate(path):
            val = _not_found_
            if key is None:
                val = self
            elif self is _not_found_:
                break
            elif isinstance(self, Entry):
                return self.move_to(path[idx:-1], _not_found_),  path[-1]
            # elif isinstance(self, EntryContainer):
            #     return self.get(path[idx:-1], _not_found_), path[-1]
            elif isinstance(self, array_type) and isinstance(key, (int, slice)):
                try:
                    val = self[key]
                except (IndexError, KeyError, TypeError) as error:
                    logger.exception(error)
                    val = _not_found_
            elif isinstance(self, (collections.abc.Mapping)) and isinstance(key, str):
                val = self.get(key, _not_found_)
            elif not isinstance(self, (collections.abc.Sequence)):
                raise NotImplementedError(f"{type(self)} {type(key)} {path[:idx+1]}")
            elif key is _next_:
                self.append(_not_found_)
                key = len(self)-1
                val = _not_found_
            elif key is _last_:
                val = self[-1]
                key = len(self)-1
            elif isinstance(key, (int, slice)):
                try:
                    val = self[key]
                except (IndexError, KeyError, TypeError) as error:
                    # logger.exception(error)
                    val = _not_found_
            elif isinstance(key, dict):
                iv_list = [[i, v] for i, v in enumerate(
                    self) if Entry._match(v, predication=key)]
                if len(iv_list) == 0:
                    if force:
                        val = deepcopy(key)
                        self.append(val)
                        key = len(self)-1
                elif len(iv_list) == 1:
                    key, val = iv_list[0]
                else:
                    key = [i for i, v in iv_list]
                    val = [v for i, v in iv_list]
                    if any(filter(lambda d:  isinstance(d, Entry), val)):
                        val = EntryCombiner(val, path[idx+1:])
            else:
                val = [Entry._eval_path(d, key, force=force) for d in self]

            if idx < last_index:
                if val is _not_found_:
                    if force:
                        val = _DICT_TYPE_() if isinstance(
                            path[idx+1], str) else _LIST_TYPE_()
                        self[key] = val
                    else:
                        key = path[idx:]
                        break
                self = val

        # if self is _not_found_:
        #     raise KeyError((path, self))
        return self, key

    @staticmethod
    def _eval_filter(self: _T, predication=_undefined_, only_first=False) -> _T:
        if not isinstance(self, list) or predication is _undefined_:
            return [self]
        if only_first:
            try:
                val = next(
                    filter(lambda d: Entry._match(d, predication), self))
            except StopIteration:
                val = _not_found_
            else:
                val = [val]
        else:
            val = [d for d in self if Entry._match(d, predication)]

        return val

    @staticmethod
    def _eval_pull(self, path: list, query=_undefined_, *args, lazy=False):
        """
            if path is found then
                return value
            else
                if lazy then return Entry(self,path) else return _not_found_
        """
        if isinstance(self, Entry):
            return self.get(path, default_value=_not_found_, *args, query=query, lazy=lazy)
        # elif isinstance(self, EntryContainer):
        #     return self.get(path, default_value=_not_found_, *args,  query=query, lazy=lazy)

        self, key = Entry._eval_path(self, path+[None], force=False)

        if any(filter(lambda d: isinstance(d, dict), path)):
            if not isinstance(self, list):
                pass
            elif len(self) == 1:
                self = self[0]
            elif len(self) == 0:
                self = _not_found_

        if query is _undefined_ or query is _not_found_ or query is None:
            if key is None:
                return self
            elif lazy is True:
                return Entry(self, key[:-1])
            else:
                return _not_found_

        if key is not None:
            self = _not_found_

        if isinstance(query, str) and query[0] == '@':
            query = Entry.op_tag.__members__[query[1:]]
            val = Entry._ops[query](self, *args)
        elif isinstance(query, Entry.op_tag):
            val = Entry._ops[query](self, *args)
        elif query is None or query is _undefined_:
            val = self
        elif not isinstance(query, dict):
            raise NotImplementedError(query)
            # val, key = Entry._eval_path(self, Entry.normalize_path(query)+[None], force=False)
            # if key is not None:
            #     val = query
        else:
            val = {k: Entry._eval_pull(self, Entry.normalize_path(k), v, *args)
                   for k, v in query.items() if not isinstance(k, Entry.op_tag)}
            if len(val) == 0:
                val = [Entry._ops[op](self, v, *args)
                       for op, v in query.items() if isinstance(op, Entry.op_tag)]

                if len(val) == 1:
                    val = val[0]
                elif len(val) == 0:
                    val = _not_found_

        return val

    @staticmethod
    def _eval_push(self, path: list, value=_undefined_, *args):
        if isinstance(self, Entry):
            return self.push(path, value, *args)
        # elif isinstance(self, EntryContainer):
        #     return self.put(path,  value, *args)

        if path is _undefined_:
            path = []
        elif not isinstance(path, list):
            path = [path]
        if not isinstance(value, array_type) and value is _undefined_:
            val = value
        elif isinstance(value, dict):
            self, p = Entry._eval_path(self, path+[""], force=True)
            if p != "":
                raise KeyError(path)
            val_changed = [Entry._eval_push(self, [k], v, *args)
                           for k, v in value.items() if not isinstance(k, Entry.op_tag)]
            val = [Entry._ops[op](self, v, *args)
                   for op, v in value.items() if isinstance(op, Entry.op_tag)]

            # if len(val) == 1:
            #     val = val[0]
            val = self
        else:
            self, p = Entry._eval_path(self, path, force=True)
            if self is _not_found_:
                raise KeyError(path)
            if isinstance(value, Entry.op_tag):
                val = Entry._ops[value](self, p, *args)
            elif isinstance(value, str) and value[0] == '@':
                value = Entry.op_tag.__members__[value[1:]]
                val = Entry._ops[value](self, p, *args)
            elif isinstance(self, Entry):
                val = self.put([p], value)
            elif isinstance(self, EntryContainer):
                val = self.put([p],  value)
            elif isinstance(self, list) and isinstance(p, int):
                val = value
                if p >= len(self):
                    self.extend([None]*(p-len(self)+1))
                self[p] = val
            else:
                val = value
                try:
                    self[p] = val
                except (KeyError, IndexError) as error:
                    logger.exception(error)
                    raise KeyError(path)

        return val

    def _push(self, path, value, predication=_undefined_, only_first=False) -> _T:
        path = self._path / path

        if self._cache is _not_found_ or self._cache is _undefined_ or self._cache is None:
            if len(path) > 0 and isinstance(path[0], str):
                self._cache = _DICT_TYPE_()
            else:
                self._cache = _LIST_TYPE_()

        if predication is _undefined_:
            self, key = Entry._eval_path(self._cache, path, force=True)

            if self is _not_found_ or isinstance(key, list):
                raise KeyError(path)
            val = Entry._eval_push(self, [key] if key is not None else [], value)
        else:
            self, key = Entry._eval_path(self._cache, path+[None], force=True)
            if key is not None or self is _not_found_:
                raise KeyError(path)
            elif not isinstance(self, list):
                raise TypeError(f"If predication is defined, self must be list! {type(self)}")
            elif only_first:
                try:
                    self = next(filter(lambda d: Entry._match(d, predication), self))
                except StopIteration:
                    val = _not_found_
                else:
                    val = Entry._eval_push(self, [], value)
            else:
                val = [Entry._eval_push(d, [], value) for d in self if Entry._match(d, predication)]
                if len(val) == 0:
                    val = _not_found_

        return val

    def replace(self, path, value: _T, **kwargs) -> _T:
        if isinstance(value, EntryContainer) and value._entry._cache is self._cache:
            value.flush()
        return self.push(path, value, **kwargs)

    def get(self, path, default_value=_undefined_, *args, lazy=False, **kwargs) -> Any:

        obj = self.find(path, *args, lazy=lazy, **kwargs)
        if obj is not _not_found_:
            return obj
        elif lazy is True and default_value is _undefined_:
            return copy(self).move_to(path)
        elif default_value is not _undefined_:
            return default_value
        else:
            raise KeyError(path)

    def put(self, *args, **kwargs) -> typing.Any:
        return self.push(*args, **kwargs)

    def get_many(self, key_list) -> typing.Mapping:
        return {key: self.get(key, None) for key in key_list}

    def dump(self, *args, **kwargs):
        """
            convert data in cache to python native type and array_type
            [str, bool, float, int, array_type, Sequence, Mapping]:
        """
        return as_native(self._cache, *args, **kwargs)

    def pull_(self, path=None, query=_undefined_,  lazy=False, predication=_undefined_, only_first=False, type_hint=_undefined_) -> Any:
        if isinstance(path, (Entry.op_tag)) and query is _undefined_:
            query = path
            path = None

        path = self._path+Entry.normalize_path(path)

        if predication is _undefined_:
            val = Entry._eval_pull(self._cache, path, query, lazy=lazy)
        else:
            self, key = Entry._eval_path(
                self._cache, path+[None], force=False)
            if key is not None:
                val = Entry._eval_pull(_not_found_, [],  query)
            elif not isinstance(self, list):
                raise TypeError(
                    f"If predication is defined, self must be list! {type(self)}")
            elif only_first:
                try:
                    self = next(
                        filter(lambda d: Entry._match(d, predication), self))
                except StopIteration:
                    self = _not_found_
                val = Entry._eval_pull(self, [],  query)
            else:
                val = [Entry._eval_pull(d, [],  query)
                       for d in self if Entry._match(d, predication)]

        return val

    def push_(self, target, op, *args, **kwargs) -> bool:
        if (self._path is None or self._path.empty):
            if self._cache is None or op in (Entry.op_tags.assign, Entry.op_tags.null):
                self._cache = value
                return value
            else:
                return normal_put(self._cache, _undefined_, value, op)
        else:
            if self._cache is not None:
                pass
            elif isinstance(self._path[0], str):
                self._cache = {}
            else:
                self._cache = []

            return normal_put(self._cache, self._path, value, op)


def as_path(path):
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path


# 测试代码
