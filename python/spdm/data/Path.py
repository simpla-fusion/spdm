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
class OpTags(Flag):
    # traversal operation 操作
    root  = auto()  # root node
    parent = auto()  # parent node
    current = auto()  # current node
    next  = auto()  # next sibling

    # RESTful operation for CRUD
    fetch  = auto()  # GET
    update = auto()  # PUT
    insert = auto()  # POST
    remove = auto()  # DELETE

    call  = auto()  # call function
    exists = auto()
    check_type = auto() # check type
    search = auto()  # search by query return idx
    dump  = auto()   # rescurive get all data

    # for sequence
    reduce = auto()

    sort  = auto()

    # predicate 谓词
    check  = auto()
    count  = auto()

    # boolean
    equal  = auto()
    le   = auto()
    ge   = auto()
    less  = auto()
    greater = auto()
# fmt:on


class Query:
    def __init__(self, query: dict | None = None, **kwargs) -> None:
        if query is not None and not isinstance(query, dict):
            raise TypeError(f"Query only support OpTags, str or dict, not {type(query)}!")

        self._query = Query._parser(merge_tree_recursive(query, kwargs))

    def __str__(self) -> str: return str(self._query)

    @staticmethod
    def _parser(query: dict, **kwargs) -> dict:

        if query is None:
            query = {".": Path._op_fetch}

        elif isinstance(query, Path.tags):
            query = {".": f"${query.name}"}

        elif isinstance(query, str) and query.startswith("$"):
            query = {".": query}

        elif isinstance(query, dict):
            query = {k: Query._parser(v) for k, v in query.items()}

        elif not isinstance(query, str):
            raise TypeError(f"{type(query)}")

        return query

    def __call__(self, target, *args, **kwargs) -> typing.Any:
        return self.check(target, *args, **kwargs)

    @staticmethod
    def _eval_one(target, k, v) -> typing.Any:

        if k == "." or k is OpTags.current:
            return Query._q_equal(target, v)

        elif isinstance(k, str) and k.startswith("@"):
            return Query._q_equal(getattr(target, k[1:], _not_found_), v) or Query._q_equal(target.get(k[1:], _not_found_), v)

        return True

    def _eval(self, target) -> bool:
        return all([Query._eval_one(target, k, v) for k, v in self._query.items()])

    def check(self, target) -> bool:
        res = self._eval(target)

        if isinstance(res, list):
            return all(res)
        else:
            return bool(res)

    def find_next(self, target, start: int | None, **kwargs) -> typing.Tuple[typing.Any,  int | None]:
        next = start
        return _not_found_, next

    @staticmethod
    def _q_equal(target, value) -> bool:
        if isinstance(target, collections.abc.Sequence):
            return value in target
        else:
            return target == value

    # fmt: off
    _q_neg         =np.negative   
    _q_add         =np.add     
    _q_sub         =np.subtract   
    _q_mul         =np.multiply   
    _q_matmul      =np.matmul    
    _q_truediv     =np.true_divide 
    _q_pow         =np.power    
    _q_equal       =np.equal    
    _q_ne          =np.not_equal  
    _q_lt          =np.less     
    _q_le          =np.less_equal  
    _q_gt          =np.greater   
    _q_ge          =np.greater_equal
    _q_radd        =np.add     
    _q_rsub        =np.subtract   
    _q_rmul        =np.multiply   
    _q_rmatmul     =np.matmul    
    _q_rtruediv    =np.divide    
    _q_rpow        =np.power    
    _q_abs         =np.abs     
    _q_pos         =np.positive   
    _q_invert      =np.invert    
    _q_and         =np.bitwise_and 
    _q_or          =np.bitwise_or  
    _q_xor         =np.bitwise_xor 
    _q_rand        =np.bitwise_and 
    _q_ror         =np.bitwise_or  
    _q_rxor        =np.bitwise_xor 
    _q_rshift      =np.right_shift 
    _q_lshift      =np.left_shift  
    _q_rrshift     =np.right_shift 
    _q_rlshift     =np.left_shift  
    _q_mod         =np.mod     
    _q_rmod        =np.mod     
    _q_floordiv    =np.floor_divide 
    _q_rfloordiv_  =np.floor_divide 
    _q_trunc       =np.trunc    
    _q_round       =np.round    
    _q_floor       =np.floor    
    _q_ceil        =np.ceil     


    # fmt: on

PathLike = int | str | slice | typing.Dict | typing.List | OpTags | None

path_like = (int, str, slice, list, None, tuple, set, dict, OpTags)


class PathError(Exception):
    def __init__(self, path: typing.List[PathLike], message: str | None = None) -> None:
        if message is None:
            message = f"PathError: {Path(path)}"
        else:
            message = f"PathError: {Path(path)}: {message}"
        super().__init__(message)


class Path(list):
    """
    Path用于描述数据的路径, 在 HTree ( Hierarchical Tree) 中定位Element, 其语法是 JSONPath 和 XPath的变体，
    并扩展谓词（predicate）语法/查询选择器。

    HTree: Hierarchical Tree 半结构化树状数据，树节点具有 list或dict类型，叶节点为 list和dict 之外的primary数据类型，
    包括 int，float,string 和 ndarray。

    基本原则是用python 原生数据类型（例如，list, dict,set,tuple）等

    DELIMITER=`/` or `.`

    | Python 算符          | 字符形式          | 描述
    | ----             |---            | ---
    | N/A              | `$`            | 根对象 （ TODO：Not Implemented ）
    | None             | `@`            | 空选择符，当前对象。当以Path以None为最后一个item时，表示所指元素为leaf节点。
    | `__truediv__`,`__getattr___` | DELIMITER (`/` or `.`)  | 子元素选择符, DELIMITER 可选
    | `__getitem__`         | `[index|slice|selector]`| 数组元素选择符，index为整数,slice，或selector选择器（predicate谓词）

    predicate: 谓词, 过滤表达式，用于过滤数组元素.
    | `set`             | `[{a,b,1}]`        | 返回dict, named并集运算符，用于组合多个子元素选择器，并将element作为返回的key， {'a':@[a], 'b':@['b'], 1:@[1] }
    | `list`            | `["a",b,1]`        | 返回list, 并集运算符，用于组合多个子元素选择器，[@[a], @['b'], @[1]]
    | `slice`            | `[start:end:step]`，   | 数组切片运算符, 当前元素为 ndarray 时返回数组切片 @[<slice>]，当前元素为 dict,list 以slice选取返回 list （generator），
    | `slice(None) `        | `*`            | 通配符，匹配任意字段或数组元素，代表所有子节点（children）
    |                | `..`           | 递归下降运算符 (Not Implemented)
    | `dict` `{$eq:4, }`      | `[?(expression)]`     | 谓词（predicate）或过滤表达式，用于过滤数组元素.
    |                | `==、!=、<、<=、>、>=`   | 比较运算符

    examples：
    | Path               | Description
    | ----               | ---
    | `a/b/c`              | 选择a节点的b节点的c节点
    | `a/b/c/1`             | 选择a节点的b节点的c节点的第二个元素
    | `a/b/c[1:3]`           | 选择a节点的b节点的c节点的第二个和第三个元素
    | `a/b/c[1:3:2]`          | 选择a节点的b节点的c节点的第二个和第三个元素
    | `a/b/c[1:3:-1]`          | 选择a节点的b节点的c节点的第三个和第二个元素
    | `a/b/c[d,e,f]`          |
    | `a/b/c[{d,e,f}]          |
    | `a/b/c[{value:{$le:10}}]/value  |
    | `a/b/c.$next/           |
    主要的方法：
    find
    """
    delimiter = "/"
    tags = OpTags

    def __init__(self, path=None, **kwargs):
        super().__init__(Path._parser(path), **kwargs)

    def __repr__(self): return Path._to_str(self)

    def __str__(self): return Path._to_str(self)

    def __hash__(self) -> int: return self.__str__().__hash__()

    def __copy__(self) -> Path: return self.__class__(self[:])

    def as_url(self) -> str: return Path._to_str_decprecated(self)

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

    def prepend(self, d) -> Path:
        res = as_path(d)
        return res.append(self)

    def append(self, d) -> Path:
        if isinstance(d, str):
            d = Path._parser_str(d)
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
        #   res = [res]

        return res

    # example:
    # a/b_c6/c[{value:{$le:10}}][value]/D/[1，2/3，4，5]/6/7.9.8
    PATH_REGEX = re.compile(r"(?P<key>[^\[\]\/\,\.]+)|(\[(?P<selector>[^\[\]]+)\])")

    # 正则表达式解析，匹配一段被 {} 包裹的字符串
    PATH_REGEX_DICT = re.compile(r"\{(?P<selector>[^\{\}]+)\}")

    @staticmethod
    def _to_str_decprecated(p: typing.Any) -> str:

        if isinstance(p, list):
            return Path.delimiter.join(map(Path._to_str_decprecated, p))
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
    def _from_str_decprecated(path: str) -> list:
        """
        """

        path_res = []

        path_list: list = path.split(Path.delimiter)

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
    def _unroll_decprecated(path: PathLike | Path.tags) -> list:
        """ Parse the to list """

        if path is None:
            return []
        elif not isinstance(path, list):
            path = [path]

        res = []

        for p in path:
            if isinstance(p, str):
                res.extend(Path._from_str_decprecated(p))
            elif isinstance(p, list):
                res.extend(Path._unroll_decprecated(p))

            else:
                res.append(p)
        return res

    @staticmethod
    def _unroll(source: typing.List[PathLike], target: typing.List[PathLike]) -> typing.List[PathLike]:
        """ Parse the to list """

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
    def _to_str(p: typing.Any) -> str:

        if isinstance(p, list):
            return Path.delimiter.join([Path._to_str(s) for s in p])
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
            m_str = ','.join([Path._to_str(s) for s in p])
            return f"({m_str})"
        elif isinstance(p, set):
            m_str = ','.join([Path._to_str(s) for s in p])
            return f"{{{m_str}}}"
        elif p is None:
            return ""
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    @staticmethod
    def _parser_decprecated(path: PathLike | Path.tags) -> list:
        path = Path._unroll_decprecated(path)
        return path

    @staticmethod
    def _parser_str_one(s: str | list) -> list | dict | str | int | slice | Path.tags:
        if isinstance(s, str):
            s = s.strip(" ")

        if not isinstance(s, str):
            item = s
        elif s.startswith(("[", "(", "{")) and s.endswith(("}", ")", "]")):
            tmp = ast.literal_eval(s)
            if isinstance(tmp, dict):
                item = {Path._parser_str_one(k): d for k, d in tmp.items()}
            elif isinstance(tmp, set):
                item = set([Path._parser_str_one(k) for k in tmp])
            elif isinstance(tmp, tuple):
                item = tuple([Path._parser_str_one(k) for k in tmp])
            elif isinstance(tmp, list):
                item = [Path._parser_str_one(k) for k in tmp]

        elif s.startswith("(") and s.endswith(")"):
            tmp: dict = ast.literal_eval(s)
            item = {Path._parser_str_one(k): d for k, d in tmp.items()}
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
    def _parser_str(path: str | list) -> list:
        """ Parse the path string to list """

        if isinstance(path, str):
            path = path.split(Path.delimiter)
            if path[0] == '':
                path[0] = Path.tags.root
        elif not isinstance(path, list):
            path = [path]

        return [Path._parser_str_one(v) for v in path]

    @staticmethod
    def _parser_dict(query: dict, **kwargs):
        res = {}
        for k, v in query.items():
            if not k.startswith("$"):
                pass
            else:
                res[k] = Path._parser(v, **kwargs)
        return Query(query, **kwargs)

    @staticmethod
    def _parser(path: PathLike) -> list:
        """ Parse the PathLike to list """

        if path is None:
            path = []

        elif isinstance(path, str):
            path = Path._parser_str(path)

        elif isinstance(path, list):
            path = sum([(Path._parser_str(p) if isinstance(p, str) else [Path._parser(p)])
                        for p in path], [])

        elif isinstance(path, dict):
            path = [Path._parser_dict(path)]

        else:
            path = [path]

        return Path._unroll(path, [])

    ###########################################################
    # RESTful API:

    # 非幂等

    def insert(self, target: typing.Any, value: typing.Any, *args, quiet=True,  **kwargs) -> typing.Tuple[typing.Any, Path]:
        """
          根据路径（self）向 target 添加元素。
          当路径指向位置为空时，创建（create）元素
          当路径指向位置为 list 时，追加（ insert ）元素
          当路径指向位置为非 list 时，合并为 [old,new]
          当路径指向位置为 dict, 添加值亦为 dict 时，根据 key 递归执行 insert

          返回新添加元素的路径

          对应 RESTful 中的 post，非幂等操作
        """
        # target, next_id = Path._exec(target, self[:-1], Path.tags.insert,
        #                              self[-1] if len(self) > 0 else None,  value, *args, quiet=quiet, **kwargs)
        # return target, Path(self[:-1]+[next_id])

        root = {"_": target}

        path = ["_"] + self[:]

        parent = Path._make_path(root, path, quiet=quiet)

        new_pos = Path._op_insert(parent,   path[-1], value, *args, **kwargs)

        if new_pos is not None:
            path += [new_pos]

        return root["_"], Path(path[1:])

    # 幂等
    def remove(self, target: typing.Any, *args,  **kwargs) -> typing.Tuple[typing.Any, int]:
        """ 根据路径（self）删除 target 中的元素。

          if quiet is False then raise KeyError if the path is not found

          对应 RESTful 中的 delete， 幂等操作

          返回修改后的target和删除的元素的个数
        """
        # target, num = Path._exec(target, self[:-1], Path.tags.remove, self[-1]
        #                          if len(self) > 0 else None, *args, quiet=quiet, **kwargs)
        # return target, num
        root = {"_": target}

        path = ["_"] + self[:]

        parent, suffix = Path._get_by_path(root, path[:-1], default_value=_not_found_)

        if len(suffix) > 0:
            return target, 0
        else:
            return root.get("_", _not_found_), Path._op_remove(parent, path[-1])

    def update(self, target: typing.Any, value: typing.Any, *args, quiet=True,   **kwargs) -> typing.Any:
        """
          根据路径（self）更新 target 中的元素。
          当路径指向位置为空时，创建（create）元素
          当路径指向位置为 dict, 添加值亦为 dict 时，根据 key 递归执行 update
          当路径指向位置为空时，用新的值替代（replace）元素

          对应 RESTful 中的 put， 幂等操作

          返回修改后的target
        """
        # target = Path._exec(target, self[:-1], Path.tags.update,
        #                     self[-1] if len(self) > 0 else None,  value, *args, quiet=quiet, **kwargs)
        # return target

        root = {"_": target}

        path = ["_"]+self[:]

        parent = Path._make_path(root, path, quiet=quiet)

        Path._op_update(parent, path[-1], value, *args, **kwargs)

        return root["_"]

    def fetch(self, target: typing.Any, op: Path.tags | str | None = None, *args,  **kwargs) -> typing.Any:
        """
            根据路径（self）查询元素。
            只读，不会修改 target

            对应 RESTful 中的 read，
            幂等操作
        """
        obj, suffix = Path._get_by_path(target, self[:])

        if len(suffix) > 0:
            obj = _not_found_

        return Path._apply_op(obj, op,  *args, **kwargs)

    def find_next(self, target: typing.Any, *starts: int | None) -> typing.Tuple[typing.Any, typing.List[int | None]]:
        """ 从 start 开始搜索符合 path 的元素，返回第一个符合条件的元素和其路径。"""

        if len(self) == 0:
            path = [slice(None)]
        else:
            path = self[:]

        obj = target

        next_ids = []

        suffix = []

        for pth_pos, q in enumerate(path):
            if not isinstance(q, (Query, slice)):
                tmp = Path._op_fetch(obj, q, default_value=_not_found_)

            if isinstance(q, slice):
                tmp, next_id = Path._op_next(obj, q, *starts)

                if next_id is None:
                    break
                else:
                    next_ids.append(next_id)
            elif isinstance(q, Query):
                tmp, next_id = q.find_next(obj, *starts)

                next_ids.append(next_id)

            if tmp is _not_found_:
                suffix = self[pth_pos:]
                break

            obj = tmp

        if len(suffix) > 0:
            return _not_found_, []
        else:
            return obj, next_ids

    def for_each(self, target,  **kwargs) -> typing.Generator[typing.Any, None, None]:
        if len(self) == 0:
            query = slice(None)
        else:
            query = self[0]

        if not isinstance(query, (slice, dict)):
            raise PathError(self[:], f"Not a generator! ")

        if isinstance(query, slice):
            start = query.start if query.start is not None else 0
            stop = query.stop
            step = query.step if query.step is not None else 1
            query = None

            if stop is not None and (stop-start)*step <= 0:
                raise PathError(self[:], f"Out of range:!")
        elif isinstance(query, dict):
            start = 0
            stop = None
            step = 1
        else:
            raise ValueError(f"Illegal query! {self[0]}")

        value = target
        next_id = start

        while True:
            if stop is not None and next_id >= stop:
                break

            if query is None or Path._op_check(value, query):
                value, suffix = Path._get_by_path(target, [next_id]+self[1:],
                                                  * args, default_value=_not_found_, **kwargs)
            else:
                value = _not_found_
                suffix = []

            if value is _not_found_ or len(suffix) > 0 or (hasattr(value.__class__, "__entry__") and not value.exists):
                if stop is None:
                    break
                else:
                    yield _not_found_
            else:
                yield value

            next_id += step

    # End API
    ###########################################################
    @staticmethod
    def _apply_op(obj: typing.Any,  op: Path.tags | str | None,  *args, **kwargs):

        if op is None:
            op = Path._op_fetch
        elif isinstance(op, Path.tags):
            op = op.name
        elif isinstance(op, str) and op.startswith("$"):
            op = op[1:]

        if callable(op):
            _op = op
        elif isinstance(op, str):
            _op = getattr(Path, f"_op_{op}", None)
        else:
            _op = None

        if not callable(_op):
            raise RuntimeError(f"Can not find callable operator {op}!")

        try:
            res = _op(obj, *args,   **kwargs)
        except Exception as error:
            raise RuntimeError(f"Illegal operator \"{op}\"!") from error

        return res

    @staticmethod
    def _make_path(target: dict | list, path: typing.List[PathLike], quiet=True) -> typing.Any:

        length = len(path)

        obj = target

        pos = 0

        while pos < length-1:

            if obj is _not_found_ or obj is None:
                raise PathError(path[:pos], f"Can not find {type(target)}!")
            elif hasattr(obj.__class__, "__entry__"):
                obj = obj.__entry__.child(path[pos:length-1])
                pos = length-1
                break

            p = path[pos]

            if not isinstance(p, (int, str)):
                raise PathError(path[:pos+1])

            tmp = Path._op_fetch(obj, p, default_value=_not_found_)

            if tmp is not _not_found_:
                obj = tmp
                pos += 1
            elif quiet:
                default_value = {} if isinstance(path[pos+1], str) else []

                Path._op_update(obj, p, default_value)

                obj = Path._op_fetch(obj, p)
                pos += 1
            else:
                raise PathError(path[:pos+1])

        return obj

    @staticmethod
    def _get_by_path(target: typing.Any, path: typing.List[PathLike], *args, default_value: typing.Any = _not_found_,   **kwargs) -> typing.Any:

        length = len(path)

        obj = target

        pos = 0

        while pos < length:

            if obj is _not_found_ or obj is None:
                break

            elif hasattr(obj.__class__, "__entry__"):
                obj = obj.__entry__.child(path[pos:length-1])
                pos = length
                break

            p = path[pos]

            if not isinstance(p, (int, str)):
                break

            tmp = Path._op_fetch(obj, p, default_value=_not_found_ if pos < length-1 else default_value)

            if tmp is _not_found_:
                break

            obj = tmp

            pos += 1

        return obj, path[pos:]

    @staticmethod
    def _op_fetch(target: typing.Any,  key: int | str | None = None, *args, default_value=_not_found_, **kwargs) -> typing.Any:

        if hasattr(target.__class__, "__entry__"):
            return target.__entry__.child(key).query(*args, default_value=default_value, **kwargs)

        if isinstance(key, list):
            if len(key) == 0:
                key = _not_found_
            elif len(key) == 1:
                key = key[0]
            else:
                target = _not_found_
                key = _not_found_

        if key is _not_found_ or key is None:
            res = target

        elif isinstance(target, array_type) and isinstance(key, (int, slice, tuple)):
            res = target[key]

        elif isinstance(key, str):
            res = getattr(target, key, _not_found_)
            if res is _not_found_ and hasattr(target.__class__, "get"):
                res = target.get(key, _not_found_)
            if res is _not_found_ and hasattr(target.__class__, "__getitem__"):
                try:
                    res = target[key]
                except Exception:
                    res = _not_found_

        elif isinstance(key, int):
            if not isinstance(target, list):
                raise TypeError(f"{type(target)}")

            if key < 0:
                key += len(target)

            if key < len(target):
                res = target[key]
            else:
                res = _not_found_
            # try:
            #     res = target[key]
            # except Exception as error:
                # raise RuntimeError(f"Error: {target}[{key}]") from error

        elif isinstance(target, collections.abc.Sequence) and isinstance(key, (int, slice)):
            res = target[key]

        elif isinstance(key, set):
            res = {p: Path(p).fetch(target, *args, **kwargs) for p in key}

        elif isinstance(key, Path.tags):
            res = Path._apply_op(target, key, *args, **kwargs)

        else:
            raise NotImplementedError(f"Not implemented query! {target} '{key}'")

        # if len(args) > 0:
        #     res = Path._op_fetch(res, *args, **kwargs)

        if res is _not_found_:
            res = default_value

        return res

    @staticmethod
    def _op_dump(target: typing.Any, *args, **kwargs) -> typing.Any:
        return serialize(target)

    @staticmethod
    def _op_update(target: typing.Any, key: int | str | None, value: typing.Any, *args, **kwargs) -> typing.Any:

        if hasattr(target.__class__, "__entry__"):
            return target.__entry__.child(key).update(value, *args, **kwargs)

        elif value is _not_found_:
            return target

        elif key is None:
            if isinstance(target, (dict)) and isinstance(value, dict):
                for k, v in value.items():
                    Path._op_update(target, k, v, *args, **kwargs)
            else:
                target = value

        elif isinstance(key, str):
            if target is _not_found_:
                target = {}
            elif not isinstance(target, dict):
                raise ValueError(f"Can not insert {key} into {target}")
            target[key] = Path._op_update(target.get(key, _not_found_), None, value)

        elif isinstance(key, int):
            if target is _not_found_:
                target = [_not_found_]*(key+1)
            elif not isinstance(target, list):
                target = [target]+[_not_found_]*(key)
            else:
                if key < 0:
                    key += len(target)
                if key > len(target):
                    target += [_not_found_]*(key-len(target)+1)
            target[key] = Path._op_update(target[key], None, value)

        else:
            raise NotImplementedError(f"Not implemented key! '{key}'")

        return target

    @staticmethod
    def _op_insert(target: typing.Any, key: int | str | None, value: typing.Any, *args, **kwargs) -> typing.Tuple[typing.Any, int | str | None]:
        if hasattr(target.__class__, "__entry__"):
            return target.__entry__.child(key).insert(value, *args, **kwargs)

        elif value is _not_found_:
            return target

        elif key is None:

            if target is _not_found_:
                target = value
            elif not isinstance(target, (list, dict)):
                target = [target, value]
                key = 1
            elif isinstance(target, list):
                key = len(target)
                if isinstance(value, list):
                    target.extend(value)
                else:
                    target.append(value)

            elif isinstance(target, dict):
                if not isinstance(value, dict):
                    target = value
                else:
                    for k, v in value.items():
                        Path._op_insert(target, k, v, *args, **kwargs)

        elif isinstance(key, int):
            if target is _not_found_:
                target = [_not_found_]*(key+1)
            elif not isinstance(target, list):
                target = [target]+[_not_found_]*(key)
            elif key > len(target):
                target += [_not_found_]*(key-len(target)+1)
            target[key], _ = Path._op_insert(target[key], None, value)
        elif isinstance(key, str):
            if target is _not_found_:
                target = {}
            elif not isinstance(target, dict):
                raise ValueError(f"Can not insert {key} into {target}")
            target[key], _ = Path._op_insert(target.get(key, _not_found_), None, value)

        else:
            raise NotImplementedError(f"Not implemented key! '{key}'")

        return target, key

    @staticmethod
    def _op_remove(target: typing.Any, key: int | str | None, *args, **kwargs) -> typing.Tuple[typing.Any, int]:
        if len(args)+len(kwargs) > 0:
            logger.warning(f"Ignore {args} {kwargs}")

        if isinstance(key, (str, int, slice)):
            try:
                del target[key]
            except Exception as error:
                success = False
            else:
                success = True
        else:
            raise NotImplementedError(f"{key}")
        return target, 1

    @staticmethod
    def _op_check(target: typing.Any,  query, *args, **kwargs) -> bool:
        if query is None:
            return True
        # elif not isinstance(target, collections.abc.Mapping):
        #     raise TypeError(type(target))
        elif not isinstance(query, collections.abc.Mapping):
            raise TypeError(type(query))

        return all([target.get(k[1:], _not_found_) == v for k, v in query.items() if k.startswith("@")])

    @staticmethod
    def _op_check_type(target: typing.Any, key, tp, *args, **kwargs) -> bool:
        target = Path._op_fetch(target, key, default_value=_not_found_)
        return isinstance_generic(target, tp)

    # @staticmethod
    # def _op_equal(target: typing.Any, value, *args, **kwargs):
    #     return target == value

    @staticmethod
    def _op_count(target: typing.Any, *args,  **kwargs) -> int:
        if target is _not_found_:
            return 0
        elif not isinstance(target, collections.abc.Sequence) or isinstance(target, str):
            return 1
        else:
            return len(target)

    @staticmethod
    def _op_exists(target: typing.Any,  *args,  **kwargs) -> bool:
        return target is not _not_found_

    @staticmethod
    def _op_call(target, *args,  **kwargs) -> typing.Any:

        if suffix is not None:
            raise RuntimeError(f"Can not call {target}! {suffix}")
        elif not callable(target):
            raise ValueError(f"Not callable! {target}")

        return target(*args, **kwargs)

    @staticmethod
    def _op_next(target, query, start: int | None = None, *args, **kwargs) -> typing.Tuple[typing.Any, int | None]:

        if not isinstance(query, (slice, set, Query)):
            raise ValueError(f"query is not dict,slice! {query}")

        if target is _not_found_ or target is None:
            return _not_found_, None

        if isinstance(query, slice):
            if start is None or start is _not_found_:
                start = query.start or 0
            elif query.start is not None and start < query.start:
                raise IndexError(f"Out of range: {start} < {query.start}!")
            stop = query.stop or len(target)
            step = query.step or 1

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

        elif isinstance(query, Query):
            if start is None or start is _not_found_:
                start = 0

            stop = len(target)

            value = _not_found_

            while start < stop:
                value = target[start]
                if not Path._op_check(value, query, *args, **kwargs):
                    start += 1
                    continue
                else:
                    break

            if start >= stop:
                return _not_found_, None
            else:
                return value, start

        else:
            raise NotImplementedError(f"Not implemented yet! {type(query)}")

    @staticmethod
    def _op_search(target: typing.Any, key, query, start=None, *args, **kwargs):

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
            yield from ((k, Path._find(target, concate_path(k, path[pos+1:]), *args, **kwargs)) for k in path[pos])
        elif isinstance(path[pos], tuple):
            yield from (Path._find(target, concate_path(k, path[pos+1:]), *args, **kwargs) for k in path[pos])
        elif isinstance(path[pos], slice):
            if isinstance(target, (array_type)):
                yield from Path._find(target[path[pos]], path[pos+1:], *args, **kwargs)
            elif isinstance(target, (collections.abc.Sequence)) and not isinstance(target, str):
                for item in target[path[pos]]:
                    yield from Path._find(item, path[pos+1:], *args, **kwargs)
            elif isinstance(target, (collections.abc.Mapping)):
                target_ = {k: v for k, v in target.items() if k is not None}
                start = path[pos].start if path[pos].start is not None else 0
                stop = path[pos].stop if path[pos].start is not None else len(target_)
                step = path[pos].step if path[pos].step is not None else 1

                for k in range(start, stop, step):
                    yield from Path._find(target_[k], path[pos+1:], *args, **kwargs)
            # elif "default_value" in kwargs:
            #   yield kwargs["default_value"]
            else:
                raise TypeError(f"Cannot slice target={(target)} path=[{path[:pos]} ^, {path[pos:]}]")
        elif isinstance(path[pos], collections.abc.Mapping):
            only_first = kwargs.get("only_first", False) or path[pos].get("@only_first", True)
            if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
                for element in target:
                    if Path._match(element, path[pos]):
                        yield from Path._find(element, path[pos+1:], *args, **kwargs)
                        if only_first:
                            break
            # elif "default_value" in kwargs:
            #   yield [kwargs["default_value"]]
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
    if path is None or path is _not_found_:
        return Path()
    elif not isinstance(path, Path):
        return Path(path)
    else:
        return path
