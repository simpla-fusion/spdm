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
from ..utils.misc import serialize
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import array_type, isinstance_generic


# fmt:off
class PathOpTags(Flag):
    # traversal operation 操作
    root    = auto()    # root node
    parent  = auto()    # parent node
    current = auto()    # current node
    next    = auto()    # next sibling

    # RESTful operation for CRUD
    fetch   = auto()    # GET
    update  = auto()    # PUT
    insert  = auto()    # POST
    remove  = auto()    # DELETE
    exists  = auto()
    check_type = auto() # check type
    search  = auto()    # search by query return idx
    dump   = auto()     # rescurive get all data

    # for sequence
    reduce  = auto()

    sort    = auto()

    # predicate 谓词
    check   = auto()
    count   = auto()

    # boolean
    equal   = auto()
    le      = auto()
    ge      = auto()
    less    = auto()
    greater = auto()
# fmt:on


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

    def collapse(self, idx=None) -> Path:
        """ 
            - 从路径中删除非字符元素，例如 slice, dict, set, tuple，int。用于从 default_value 中提取数据
            - 从路径中删除指定位置idx: 的元素

        """
        if idx is None:
            return Path([p for p in self if isinstance(p, str)])
        else:
            return Path(self[:idx]+self[idx+1:])

    ###########################################################
    # RESTful API:

    def query(self,  target: typing.Any, op: typing.Optional[Path.tags] = None, *args, quiet=False, **kwargs) -> typing.Any:
        """ 根据路径（self）查询元素。
            只读，不会修改 target

            对应 RESTful 中的 read， 幂等操作
        """
        if op is None:
            op = Path.tags.fetch
        return Path._exec(target, self[:], op, *args, quiet=quiet, **kwargs)

    def update(self, target: typing.Any, value: typing.Any, *args, quiet=True, **kwargs) -> typing.Any:
        """ 根据路径（self）更新 target 中的元素。
            当路径指向位置为空时，创建（create）元素
            当路径指向位置为 dict, 添加值亦为 dict 时，根据 key 递归执行 update
            当路径指向位置为空时，用新的值替代（replace）元素

            对应 RESTful 中的 put， 幂等操作
            返回值为更新的元素路径
        """
        return Path._exec(target, self[:], Path.tags.update, value, *args, quiet=quiet, **kwargs)

    def insert(self, target: typing.Any, value: typing.Any, *args, quiet=True, **kwargs) -> PathLike | Path:
        """ 根据路径（self）向 target 添加元素。
            当路径指向位置为空时，创建（create）元素
            当路径指向位置为 list 时，追加（ insert ）元素
            当路径指向位置为非 list 时，合并为 [old,new]
            当路径指向位置为 dict, 添加值亦为 dict 时，根据 key 递归执行 insert

            返回新添加元素的路径

            对应 RESTful 中的 post，非幂等操作
        """
        return Path._exec(target,  self[:], Path.tags.insert, value, *args,  quiet=quiet, **kwargs)

    def remove(self, target: typing.Any, *args, quiet=True, **kwargs) -> int:
        """ 根据路径（self）删除 target 中的元素。

            if quiet is False then raise KeyError if the path is not found

            对应 RESTful 中的 delete， 幂等操作
            返回实际删除的元素个数
        """
        return Path._exec(target, self[:], Path.tags.remove, *args,  quiet=quiet,  **kwargs)

    def find_next(self, target: typing.Any,  start: PathLike, *args, **kwargs) -> typing.Tuple[typing.Any, PathLike]:
        """  从 start 开始搜索符合 path 的元素，返回第一个符合条件的元素和其路径。"""
        res = Path._exec(target, self[:], Path.tags.next, start, *args, **kwargs)

        if not isinstance(res, tuple):
            return res, None
        else:
            return res

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
    def _parser(path: PathLike, delimiter="/") -> list:
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
    def _exec(target: typing.Any, path: typing.List[PathLike], op,  *args,   quiet=True, **kwargs) -> typing.Any:

        if path is None:
            path = []

        length = len(path)

        obj = target

        idx = 0
        while idx < length-1:
            p = path[idx]

            if obj is _not_found_ or obj is None:
                break

            if hasattr(obj.__class__, "__entry__"):
                obj = obj.__entry__.child(path[idx:])
                idx = length
            else:
                tmp = Path._op_fetch(obj, p)

                if tmp is not _not_found_:
                    obj = tmp
                    idx += 1
                elif quiet:
                    Path._op_insert(obj, p,  {} if isinstance(path[idx+1], str) else [])
                else:
                    obj = _not_found_
                    break

        if len(path[idx:]) > 1 and obj is not _not_found_:
            raise KeyError(f"Cannot find {Path(path[idx:])} in {target}! {args}")

        if hasattr(obj, "__entry__"):
            res = obj.__entry__.child(path[idx:]).query(*args, **kwargs)

        else:
            try:
                res = Path._apply_op(obj, op, path[idx:],  *args, **kwargs)
            except StopIteration as error:
                raise error
            except IndexError as error:
                logger.error(error)
                res = _not_found_
            except KeyError as error:
                logger.error(error)
                res = _not_found_
            except Exception as error:
                # raise RuntimeError(f"Error: path={path[:idx+1]}") from error
                res = _not_found_

        return res

    @staticmethod
    def _apply_op(target: typing.Any, op: Path.tags | str,  key: list, *args, **kwargs):

        if isinstance(op, Path.tags):
            op = op.name
        elif isinstance(op, str) and op.startswith("$"):
            op = op[1:]

        if isinstance(op, str):
            _op = getattr(Path, f"_op_{op}", None)
        elif callable(op):
            _op = op
        else:
            _op = None

        if not callable(_op):
            raise RuntimeError(f"Can not find callable operator {op}!")

        if len(key) == 0 or target is _not_found_:
            key = _not_found_
        elif len(key) == 1:
            key = key[0]
        else:
            raise RuntimeError(f"Don't know how to handle {op} {key} {target}")

        try:
            res = _op(target,  key, *args,  **kwargs)
        except StopIteration as error:
            raise error
        except Exception as error:
            raise RuntimeError(f"Illegal operator \"{op}\"!") from error

        return res

    @staticmethod
    def _op_fetch(target: typing.Any, key: PathLike, *args, **kwargs) -> typing.Any:
        if hasattr(target, "__entry__"):
            return target.__entry__.child(key).query(*args, **kwargs)

        if key is _not_found_:
            res = target

        elif isinstance(target, array_type) and isinstance(key, (int, slice, tuple)):
            res = target[key]

        elif isinstance(key, str):
            res = getattr(target, key, _not_found_)
            if res is _not_found_ and hasattr(target, "get"):
                res = target.get(key, _not_found_)
            if res is _not_found_ and hasattr(target, "__getitem__"):
                try:
                    res = target[key]
                except Exception:
                    res = _not_found_

        elif isinstance(key, int):
            try:
                res = target[key]
            except Exception as error:
                raise RuntimeError(f"Error: {target}[{key}]") from error

        elif isinstance(target, collections.abc.Sequence) and isinstance(key, (int, slice)):
            res = target[key]

        elif isinstance(key, set):
            res = {p: Path(p).query(target, *args, **kwargs) for p in key}

        elif isinstance(key, dict):
            # logger.error(f"Not implemented query!  '{key}' not in  {target}")
            # res = all([Path._op_fetch(target, *kv, *args, **kwargs) for kv in key.items()])
            res, pos = Path._op_next(target, key, 0, *args, **kwargs)

        elif isinstance(key, Path.tags):
            res = Path._apply_op(target, key, [], *args, **kwargs)

        else:
            raise NotImplementedError(f"Not implemented query! '{key}'")

        if len(args) > 0:
            res = Path._op_fetch(res, *args, **kwargs)

        if res is _not_found_:
            res = kwargs.get("default_value", _not_found_)

        return res

    @staticmethod
    def _op_update(target: typing.Any, key: PathLike, value: typing.Any, *args, **kwargs) -> typing.Any:

        if key != 0 and not key:
            key = None

        if hasattr(target, "__entry__"):
            return target.__entry__.child(key).update(value, *args, **kwargs)

        elif value is _not_found_:
            return None

        elif key is None and isinstance(value, collections.abc.Mapping):
            for k, v in value.items():
                Path._op_update(target, k, v, *args, **kwargs)

        elif isinstance(target, collections.abc.Mapping) and isinstance(key, str):
            tmp = target.get(key, _not_found_)
            if isinstance(tmp, dict):
                Path._op_update(tmp, None, value, *args, **kwargs)
            else:
                target[key] = value
        elif isinstance(target, collections.abc.MutableSequence) and isinstance(key, int):
            if key > len(target):
                raise IndexError(f"{key}> {len(target)}")
            tmp = target[key]
            if isinstance(tmp, dict):
                Path._op_update(tmp, None, value, *args, **kwargs)
            else:
                target[key] = value
        else:
            raise NotImplementedError(f"{target} {key} \"{value}\"")

    @staticmethod
    def _op_insert(target: typing.Any, key: PathLike, value: typing.Any, *args, **kwargs) -> PathLike:

        if hasattr(target, "__entry__"):
            return target.__entry__.child(key).insert(value, *args, **kwargs)

        elif value is _not_found_:
            logger.warning("Nothing to insert! key={key} ")
            return []
        elif key != 0 and not key:
            key = None
            new_path = []
            _obj = target
        elif isinstance(key, (int, str)):
            new_path = [key]
            _obj = Path._op_fetch(target, key, default_value=_not_found_)
        else:
            raise NotImplementedError(f"Not implemented query! '{key}'")

        if isinstance(_obj, list) and isinstance(value, list):
            pos = len(_obj)
            _obj.extend(value)
            new_path.append(slice(pos, len(_obj)))

        elif isinstance(_obj, list) and not isinstance(value, list):
            pos = len(_obj)
            _obj.append(value)
            new_path.append(slice(pos, len(_obj)))

        elif isinstance(_obj, dict) and isinstance(value, dict):
            for k, v in value.items():
                Path._op_insert(_obj, k, v, *args, **kwargs)

        elif key is None:
            raise KeyError(f"Cannot insert {value} to {target}!")

        elif not isinstance(_obj, list) and isinstance(value, list):
            pos = len(_obj)
            target[key] = [_obj] + value
            new_path.append(slice(1, 1+len(value)))

        elif _obj is _not_found_:
            target[key] = value

        elif value is not _not_found_:
            target[key] = [_obj, value]

        else:
            logger.warning(f"Nothing to insert! {key} ")

        return new_path

    @staticmethod
    def _op_remove(target, k, *args, **kwargs):
        if len(args)+len(kwargs) > 0:
            logger.warning(f"Ignore {args} {kwargs}")

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
    def _op_check(target, query,  *args):
        if not isinstance(target, collections.abc.Mapping):
            raise TypeError(type(target))
        elif not isinstance(query, collections.abc.Mapping):
            raise TypeError(type(query))

        return all([target.get(k[1:], _not_found_) == v for k, v in query.items() if k.startswith("@")])

    @staticmethod
    def _op_check_type(target, key, tp, *args, **kwargs) -> bool:
        target = Path._op_fetch(target, key, default_value=_not_found_)
        return isinstance_generic(target, tp)

    @staticmethod
    def _op_search(target, key, query, start=None, *args, **kwargs):

        target = Path._op_fetch(target, key)

        if start is None:
            start = 0
        stop = len(target)

        pos = None

        for idx in range(start, stop):
            if Path._op_check(target[idx], query, *args, **kwargs):
                pos = idx
                break

        return pos if pos is not None else _not_found_

    @staticmethod
    def _op_exist(target, key=_not_found_, *args, **kwargs) -> bool:
        if key is _not_found_:
            return target is not _not_found_
        elif hasattr(target, "__contains__"):
            return target.__contains__(key)
        else:
            raise TypeError(type(target))

    @staticmethod
    def _op_dump(target, key, *args, **kwargs):
        data = Path._op_fetch(target, key, default_value=kwargs.pop("default_value", _not_found_))
        return serialize(data, *args, **kwargs)

    @staticmethod
    def _op_equal(target,  key, value, *args, **kwargs):
        data = Path._op_fetch(target, key, default_value=kwargs.pop("default_value", _not_found_))
        return data == value

    @staticmethod
    def _op_count(target, key, *args, **kwargs) -> int:

        data = Path._op_fetch(target, key, default_value=kwargs.pop("default_value", _not_found_))

        if data is _not_found_:
            return 0
        elif not isinstance(data, (collections.abc.Sequence, collections.abc.Mapping)) or isinstance(data, str):
            return 1
        else:
            return len(data)

    @staticmethod
    def _op_exists(target, key, *args, **kwargs) -> bool:
        if target is _not_found_ or target is None:
            return False
        elif key is _not_found_:
            return target is not _not_found_
        elif target is _not_found_:
            return False
        elif isinstance(target, dict):
            return key in target
        elif isinstance(target, collections.abc.Sequence) and isinstance(key, int):
            return key >= 0 and key < len(target)
        else:
            raise NotImplementedError(f"{target} {key}")

    @staticmethod
    def _op_next(target, key, start: int | None, *args, **kwargs) -> typing.Tuple[typing.Any, int | None]:

        if key is _not_found_ or key is None:
            key = slice(None)
        elif not isinstance(key, (slice, dict, set)):
            target = Path._op_fetch(target, key, default_value=_not_found_)
            key = slice(None)

        if target is _not_found_ or target is None:
            return _not_found_, None
        elif not isinstance(target, collections.abc.Sequence):
            raise TypeError((target))

        if isinstance(key, slice):
            if start is None or start is _not_found_:
                start = key.start or 0
            elif key.start is not None and start < key.start:
                raise IndexError(f"Out of range: {start} < {key.start}!")
            stop = key.stop or len(target)
            step = key.step or 1

            if start >= stop:
                # raise StopIteration(f"Can not find next entry of {start}>={stop}!")
                return None, None
            else:
                value = Path._op_fetch(target, start, *args, default_value=_not_found_, **kwargs)

                if value is _not_found_:
                    start = None
                else:
                    start += step

                return value, start

        elif isinstance(key, dict):
            if start is None or start is _not_found_:
                start = 0

            stop = len(target)

            value = _not_found_

            while start < stop:
                value = target[start]
                if not Path._op_check(value, key, *args, **kwargs):
                    start += 1
                    continue
                else:
                    break

            if start >= stop:
                return _not_found_, None
            else:
                return value, start

        else:
            raise NotImplementedError(f"Not implemented yet! {type(key)}")

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

    ############################################################
    # deprecated method
    @deprecated
    def traversal(self) -> typing.Generator[PathLike, None, None]:
        yield from Path._traversal_path(self[:])

    @deprecated
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

    @deprecated
    @staticmethod
    def _traversal_path(path: typing.List[typing.Any], prefix: typing.List[typing.Any] = []) -> typing.Generator[PathLike, None, None]:
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

    @deprecated
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

    @deprecated
    @staticmethod
    def _find_all(target: typing.Any, path: typing.List[typing.Any], *args, **kwargs):
        return Path._expand(Path._find(target, path, *args, **kwargs))


def as_path(path):
    if not isinstance(path, Path):
        return Path(path)
    else:
        return path
