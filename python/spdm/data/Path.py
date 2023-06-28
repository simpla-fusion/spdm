from __future__ import annotations

import ast
import collections.abc
import pprint
import re
import typing
from copy import copy, deepcopy
from enum import Flag, auto

import numpy as np

from ..utils.logger import deprecated, experimental, logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import array_type


class PathOpTags(Flag):
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
    exists = auto()
    # boolean
    equal = auto()
    le = auto()
    ge = auto()
    less = auto()
    greater = auto()


PathLike = int | str | slice | typing.Dict | typing.List | PathOpTags | None

path_like = (int, str, slice, list, None, tuple, set, dict, PathOpTags)


class Path(list):
    """
    Path用于描述数据的路径, 在 HTree ( Hierarchical Tree) 中定位Element, 其语法是 JSONPath 和 XPath的变体，
    并扩展谓词（predicate）语法/查询选择器。

    HTree: Hierarchical Tree 半结构化树状数据，树节点具有 list或dict类型，叶节点为 list和dict 之外的primary数据类型，
    包括 int，float,string 和 ndarray。

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

    tags = PathOpTags

    def __init__(self, path=None, delimiter='/', **kwargs):
        super().__init__(Path._parser(path, delimiter=delimiter), **kwargs)
        self._delimiter = delimiter

    def __repr__(self): return Path._to_str(self, self._delimiter)

    def __str__(self): return Path._to_str(self, self._delimiter)

    def __hash__(self) -> int: return self.__str__().__hash__()

    def __copy__(self) -> Path: return self.__class__(self[:])

    def as_url(self) -> str: return Path._to_str_decprecated(self, delimiter=self._delimiter)

    @property
    def is_leaf(self) -> bool: return len(self) > 0 and self[-1] is None

    @property
    def is_root(self) -> bool: return len(self) == 0

    @property
    def is_regular(self) -> bool: return not self.is_generator

    @property
    def is_generator(self) -> bool: return any([isinstance(v, (slice, dict)) for v in self])

    @property
    def parent(self) -> Path:
        if self.is_root:
            raise RuntimeError("Root node hasn't parents")
        other = copy(self)
        other.pop()
        return other

    @property
    def children(self) -> Path:
        if self.is_leaf:
            raise RuntimeError("Leaf node hasn't child!")
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

    def append_decprecated(self, d) -> Path:

        if isinstance(d, Path):
            self.extend_decprecated(d[:])
        else:
            self.extend_decprecated(Path._parser(d))
        return self

    def extend_decprecated(self, *args, force=False) -> Path:

        if len(args) == 1:
            args = args[0]
        if force:
            super().extend(list(args))
        else:
            super().extend(Path.normalize(args))
        return self

    def append(self, d) -> Path:
        if isinstance(d, str):
            d = Path._from_str(d, delimiter=self._delimiter)
        elif not isinstance(d, collections.abc.Sequence):
            d = [d]
        return Path._unroll(d, self)

    def extend(self, d: list) -> Path: return Path._unroll(d, self)

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

    def query(self, target: typing.Any, *args, **kwargs) -> typing.Any:
        return Path._query_experimental(target, self[:], *args, **kwargs)

    def insert(self, target: typing.Any, *args, **kwargs) -> int:
        return Path._insert(target, self[:], *args, **kwargs)

    def remove(self, target: typing.Any, *args, **kwargs) -> int:
        return Path._remove(target, self[:], *args, **kwargs)

    def update(self, target: typing.Any, *args, **kwargs) -> int:
        return Path._update(target, self[:], *args,  **kwargs)

    def traversal(self) -> typing.Generator[typing.List[typing.Any], None, None]:
        yield from Path._traversal_path(self[:])

    def find(self, target: typing.Any, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        yield from Path._find(target, self[:], *args, **kwargs)

    # End API
    ###########################################################

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
            res = Path._from_str_decprecated(p)
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
    def _to_str_decprecated(p: typing.Any, delimiter=None) -> str:
        if delimiter is None:
            delimiter = Path.DELIMITER

        if isinstance(p, list):
            return delimiter.join(map(Path._to_str_decprecated, p))
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
            m_str = ','.join([f"{k}:{Path._to_str_decprecated(v)}" for k, v in p.items()])
            return f"?{{{m_str}}}"
        elif isinstance(p, tuple):
            m_str = ','.join(map(Path._to_str_decprecated, p))
            return f"({m_str})"
        elif isinstance(p, set):
            m_str = ','.join(map(Path._to_str_decprecated, p))
            return f"{{{m_str}}}"
        elif p is None:
            return ""
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    @staticmethod
    def _from_str_decprecated(path: str, delimiter=None) -> list:
        """
        """
        if delimiter is None:
            delimiter = Path.DELIMITER

        path_res = []

        path_list: list = path.split(delimiter)

        if path_list[0] == '':
            path_list[0] = Path.tags.root

        for v in path_list:
            if v.startswith(("[", "(", "{")) and v.endswith(("]", "}", ")")):
                try:
                    item = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    try:
                        item = slice(*map(int, v[1:-1].split(':')))
                    except ValueError:
                        raise ValueError(f"Invalid Path: {v}")
            elif v == "*":
                item = slice(None)
            elif v == "..":
                item = Path.tags.parent
            elif v == ".":
                continue
            elif v.isnumeric():
                item = int(v)
            elif v.startswith("$"):
                try:
                    item = Path.tags[v[1:]]
                except Exception:
                    item = v
            else:
                item = v

            path_res.append(item)

        return path_res

    @staticmethod
    def _unroll_decprecated(path: PathLike | Path.tags, delimiter=None) -> list:
        """ Parse the  to list """

        if path is None:
            return []
        elif not isinstance(path, list):
            path = [path]

        res = []

        for p in path:
            if isinstance(p, str):
                res.extend(Path._from_str_decprecated(p, delimiter=delimiter))
            elif isinstance(p, list):
                res.extend(Path._unroll_decprecated(p, delimiter=delimiter))

            else:
                res.append(p)
        return res

    @staticmethod
    def _parser_decprecated(path: PathLike | Path.tags, delimiter) -> list:
        path = Path._unroll_decprecated(path, delimiter=delimiter)
        return path

    @staticmethod
    def _to_str(p: typing.Any, delimiter) -> str:

        if isinstance(p, list):
            return delimiter.join([Path._to_str(s, delimiter) for s in p])
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
            m_str = ','.join([f"{k}:{Path._to_str(v, delimiter)}" for k, v in p.items()])
            return f"?{{{m_str}}}"
        elif isinstance(p, tuple):
            m_str = ','.join([Path._to_str(s, delimiter) for s in p])
            return f"({m_str})"
        elif isinstance(p, set):
            m_str = ','.join([Path._to_str(s, delimiter) for s in p])
            return f"{{{m_str}}}"
        elif p is None:
            return ""
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    @staticmethod
    def _from_str_one(s: str | list, delimiter) -> list | dict | str | int | slice | Path.tags:
        if isinstance(s, str):
            s = s.strip(" ")

        if not isinstance(s, str):
            item = s
        elif s.startswith(("[", "(", "{")) and s.endswith(("}", ")", "]")):
            tmp = ast.literal_eval(s)
            if isinstance(tmp, dict):
                item = {Path._from_str_one(k, delimiter): d for k, d in tmp.items()}
            elif isinstance(tmp, set):
                item = set([Path._from_str_one(k, delimiter) for k in tmp])
            elif isinstance(tmp, tuple):
                item = tuple([Path._from_str_one(k, delimiter) for k in tmp])
            elif isinstance(tmp, list):
                item = [Path._from_str_one(k, delimiter) for k in tmp]

        elif s.startswith("(") and s.endswith(")"):
            tmp: dict = ast.literal_eval(s)
            item = {Path._from_str_one(k, delimiter): d for k, d in tmp.items()}
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
    def _from_str(path: str | list, delimiter) -> list:
        """ Parse the path string to list  """

        if isinstance(path, str):
            path = path.split(delimiter)
            if path[0] == '':
                path[0] = Path.tags.root
        elif not isinstance(path, list):
            path = [path]

        return [Path._from_str_one(v, delimiter) for v in path]

    @staticmethod
    def _unroll(source: typing.List[PathLike], target: typing.List[PathLike]) -> typing.List[PathLike]:
        """ Parse the  to list """

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
    def _parser(path: PathLike, delimiter) -> list:
        if path is None:
            path = []
        elif isinstance(path, str):
            path = Path._from_str(path, delimiter)
        elif isinstance(path, list):
            path = sum([(Path._from_str(p, delimiter) if isinstance(p, str) else [p])
                        for p in path], [])
        else:
            path = [path]
        return Path._unroll(path, [])

    @staticmethod
    def _traversal(target: typing.Any, path: typing.List[typing.Any]) -> typing.Tuple[typing.Any, int]:
        """
        Traversal the target with the path, return the last regular target and the position the first non-regular path.
        :param target: the target to traversal
        :param path: the path to traversal

        """
        pos = -1
        for idx, p in enumerate(path):
            if target is None or target is _not_found_:
                break
            elif hasattr(target, "__entry__"):
                break
            elif isinstance(p, str):
                if not isinstance(target, collections.abc.Mapping):
                    tmp = getattr(target, p, _not_found_)
                    if p is _not_found_:
                        raise TypeError(f"Cannot get '{path[:idx+1]}' in {pprint.pformat(target)}")
                    else:
                        target = tmp
                        continue
                elif p not in target:
                    break
            elif isinstance(p, int):
                if isinstance(target, array_type):
                    target = target[p]
                    continue
                elif not isinstance(target, (list, tuple, collections.deque)):
                    raise TypeError(f"Cannot traversal {p} in {(target)} ")
                elif p >= len(target):
                    raise IndexError(f"Index {p} out of range {len(target)}")
            elif isinstance(p, tuple) and all(isinstance(v, (int, slice)) for v in p):
                if not isinstance(target, (array_type)):
                    break
            else:
                break
            target = target[p]
            pos = idx
        return target, pos+1

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
    def _query_decprecated(target: typing.Any, path: typing.List[typing.Any],
                           op: typing.Optional[Path.tags] = None, *args,  **kwargs) -> typing.Any:
        target, pos = Path._traversal(target, path)

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).query(op=op, *args, **kwargs)
        elif pos < len(path) - 1 and not isinstance(path[pos], (int, str)):
            res = [Path._query_decprecated(d, path[pos+1:], op, *args, **kwargs)
                   for d in Path._find(target, [path[pos]], **kwargs)]
            if len(res) == 1 and isinstance(path[pos], collections.abc.Mapping) and (all(k[0] != '@' for k in path[pos].keys()) or path[pos].get("@only_first", False)):
                res = res[0]
            elif len(res) == 0:
                res = _not_found_
        elif pos == len(path) - 1 and not isinstance(path[pos], (int, str)):
            target = Path._find_all(target, path[pos:], **kwargs)
            res = Path._query_decprecated(target, [], op, *args, **kwargs)
        elif op is None:
            if pos == len(path):
                res = target
            elif pos < len(path):  # Not found
                res = _not_found_
            else:
                res = Path._find_all(target, path[pos:], **kwargs)
        elif op == Path.tags.count:
            if pos == len(path):
                if isinstance(target, str):
                    res = 1
                else:
                    res = len(target)
            else:
                res = 0
        elif op == Path.tags.equal:
            if pos == len(path):
                res = target == args[0]
            else:
                res = False
        else:
            raise NotImplementedError(f"Not implemented yet! {op}")

        if res is _not_found_:
            res = kwargs.get("default_value", _not_found_)
        # if res is _not_found_:
        #     raise KeyError(f"Cannot find {path[:pos+1]} in {target}")
        return res

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
    def _query_experimental(target: typing.Any, path: typing.List[typing.Any],  *args, default_value=_not_found_, create_if_not_exists: typing.Any = False, **kwargs) -> typing.Any:
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
    def _update(target: typing.Any, path: typing.List[typing.Any],  *args, **kwargs) -> int:
        """
            递归合并两个 HTree

            append:
            first += second ,

            当两个字典中都有相同的键时，将这些键对应的值合并为一个列表

            - first:not list + second:list      => [first,*second]
            - first:list     + second:not list  => [*first,second]
            - first:list     + second:list      => [*first,*second]
            - first:not list + second:not list  => [first,second]
            - first:dict     + second:    dict
                for k,v in second.items():
                    first[k]= first[k] + second[k]

            update:
            first |= second ,

            当两个字典中都有相同的键时，取后者的 value

            - None        | second            => second
            - first       | None              => first
            - first:any   | second:not dict   => second
            - first:dict  | second:    dict
                for k,v in second.items():
                    first[k]= first[k] | second[k]


        """
        target, pos = Path._traversal(target, path[:-1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).update(*args, **kwargs)
        elif len(path) == 0:
            Path._update_or_replace(target, *args, **kwargs)
            return 1
        elif not isinstance(path[pos], (int, str)):
            return sum(Path._update(d, path[pos+1:],  *args, **kwargs)
                       for d in Path._find(target, [path[pos]], **kwargs))
        elif not isinstance(args[0], collections.abc.Mapping):
            return Path._insert(target, path[pos:], *args,   **kwargs)
        elif pos < len(path)-1:
            return Path._insert(target, path[pos:], Path._update_or_replace(None, *args, ),   **kwargs)
        else:  # pos == len(path)-1
            n_target, n_pos = Path._traversal(target, path[pos:])
            if n_pos == 0:
                return Path._update(n_target, path[n_pos:], Path._update_or_replace(None, *args, ),  **kwargs)

            else:
                return Path._insert(target, path[pos:], Path._update_or_replace(n_target, *args, ),  **kwargs)

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


def as_path(path):
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path
