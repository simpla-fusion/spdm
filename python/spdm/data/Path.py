from __future__ import annotations

import ast
import collections.abc
import inspect
import pprint
import re
import typing
from copy import copy, deepcopy
from enum import Flag, auto

import numpy as np

from ..utils.logger import deprecated, logger
from ..utils.misc import serialize
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import array_type, isinstance_generic, is_int


# fmt:off
class OpTags(Flag):
    # traversal operation 操作
    root  = auto()      # root node
    parent = auto()     # parent node
    children = auto()   # 所有子节点
    ancestors =auto()   # 所有祖先节点  
    descendants = auto()# 所有后代节点
    slibings=auto()     # 所有兄弟节点

    current = auto()    # current node
    next  = auto()      # next sibling
    prev  = auto()      # previous sibling

    # RESTful operation for CRUD
    find  = auto()      # GET
    update = auto()     # PUT
    insert = auto()     # POST
    remove = auto()     # DELETE
    
 

    call  = auto()      # call function
    exists = auto()
    is_leaf = auto()
    is_list=auto()
    is_dict=auto()
    check_type = auto() # check type
    keys=auto()         # 返回键值
    search = auto()     # search by query return idx
    dump  = auto()      # rescurive get all data

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


QueryLike = dict | OpTags | None


class Query:
    def __init__(self, query: QueryLike = None, only_first=True, **kwargs) -> None:
        if isinstance(query, Query):
            self._query = Query._parser(query._query, **kwargs)
        else:
            self._query = Query._parser(query, **kwargs)

        self._only_first = only_first

    def __str__(self) -> str:
        p = self._query

        if not isinstance(p, dict):
            return str(p)
        else:
            m_str = ",".join([f"{k}:{Path._to_str(v)}" for k, v in p.items()])
            return f"?{{{m_str}}}"

    @classmethod
    def _parser(cls, query: QueryLike, **kwargs) -> dict:
        if query is None:
            query = {".": Path._op_fetch}

        elif isinstance(query, Path.tags):
            query = {".": f"${query.name}"}

        elif isinstance(query, str) and query.startswith("$"):
            query = {".": query}

        elif isinstance(query, str) and query.isidentifier():
            query = {f"@{Path.id_tag_name}": query}

        elif isinstance(query, dict):
            query = {k: (Query._parser(v) if isinstance(v, dict) else v) for k, v in query.items()}

        # else:
        #     raise TypeError(f"{(query)}")

        if isinstance(query, dict):
            return update_tree(query, kwargs)

        else:
            return query

    def __call__(self, target, *args, **kwargs) -> typing.Any:
        return self.check(target, *args, **kwargs)

    @staticmethod
    def _eval_one(target, k, v) -> typing.Any:
        res = False
        if k == "." or k is OpTags.current:
            res = Query._q_equal(target, v)

        elif isinstance(k, str) and k.startswith("@"):
            if hasattr(target.__class__, k[1:]):
                res = Query._q_equal(getattr(target, k[1:], _not_found_), v)

            elif isinstance(target, collections.abc.Mapping):
                res = Query._q_equal(target.get(k, _not_found_), v)

        return res

    def _eval(self, target) -> bool:
        return all([Query._eval_one(target, k, v) for k, v in self._query.items()])

    def check(self, target) -> bool:
        res = self._eval(target)

        if isinstance(res, list):
            return all(res)
        else:
            return bool(res)

    def find_next(self, target, start: int | None, **kwargs) -> typing.Tuple[typing.Any, int | None]:
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


def as_query(query: QueryLike = None, **kwargs) -> Query | slice:
    if isinstance(query, slice):
        return query
    elif isinstance(query, Query) and len(kwargs):
        return query
    else:
        return Query(query, **kwargs)


class PathError(Exception):
    def __init__(self, path: typing.List[PathLike], message: str | None = None) -> None:
        if message is None:
            message = f"PathError: {Path(path)}"
        else:
            message = f"PathError: {Path(path)}: {message}"
        super().__init__(message)


class Path(list):
    """Path用于描述数据的路径, 在 HTree ( Hierarchical Tree) 中定位Element, 其语法是 JSONPath 和 XPath的变体，
    并扩展谓词（predicate）语法/查询选择器。

    HTree:
        Hierarchical Tree 半结构化树状数据，树节点具有 list或dict类型，叶节点为 list和dict 之外的primary数据类型，
    包括 int，float,string 和 ndarray。

    基本原则是用python 原生数据类型（例如，list, dict,set,tuple）等

    DELIMITER=`/` or `.`

    | Python 算符          | 字符形式          | 描述
    | ----             |---            | ---
    | N/A              | `$`            | 根对象 （ TODO：Not Implemented ）
    | None             | `@`            | 空选择符，当前对象。当以Path以None为最后一个item时，表示所指元素为leaf节点。
    | `__truediv__`,`__getattr___` | DELIMITER (`/` or `.`)  | 子元素选择符, DELIMITER 可选
    | `__getitem__`         | `[index|slice|selector]`| 数组元素选择符，index为整数,slice，或selector选择器（predicate谓词）

    predicate  谓词, 过滤表达式，用于过滤数组元素.

    | `set`             | `[{a,b,1}]`        | 返回dict, named并集运算符，用于组合多个子元素选择器，并将element作为返回的key， {'a':@[a], 'b':@['b'], 1:@[1] }
    | `list`            | `["a",b,1]`        | 返回list, 并集运算符，用于组合多个子元素选择器，[@[a], @['b'], @[1]]
    | `slice`            | `[start:end:step]`，   | 数组切片运算符, 当前元素为 ndarray 时返回数组切片 @[<slice>]，当前元素为 dict,list 以slice选取返回 list （generator），
    | `slice(None) `        | `*`            | 通配符，匹配任意字段或数组元素，代表所有子节点（children）
    |                | `..`           | 递归下降运算符 (Not Implemented)
    | `dict` `{$eq:4, }`      | `[?(expression)]`     | 谓词（predicate）或过滤表达式，用于过滤数组元素.
    |                | `==、!=、<、<=、>、>=`   | 比较运算符

    Examples

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

    """

    id_tag_name = "name"
    delimiter = "/"
    tags = OpTags

    def __init__(self, *args, **kwargs):
        super().__init__(args[0] if len(args) == 1 and isinstance(args[0], list) else Path._parser(args), **kwargs)

    def __repr__(self):
        return Path._to_str(self)

    def __str__(self):
        return Path._to_str(self)

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def __copy__(self) -> Path:
        return self.__class__(deepcopy(self[:]))

    def as_url(self) -> str:
        return Path._to_str(self)

    @property
    def is_leaf(self) -> bool:
        return len(self) > 0 and self[-1] is None

    @property
    def is_root(self) -> bool:
        return len(self) == 0

    @property
    def is_regular(self) -> bool:
        return not self.is_generator

    @property
    def is_generator(self) -> bool:
        return any([isinstance(v, (slice, dict)) for v in self])

    @property
    def parent(self) -> Path:
        if len(self) == 0:
            logger.warning("Root node hasn't parents")
            return self
        else:
            return Path(self[:-1])

    @property
    def children(self) -> Path:
        if self.is_leaf:
            raise RuntimeError("Leaf node hasn't child!")
        other = copy(self)
        other.append(slice(None))
        return other

    @property
    def slibings(self):
        return self._parent.children

    @property
    def next(self) -> Path:
        other = copy(self)
        other.append(Path.tags.next)
        return other

    def prepend(self, d) -> Path:
        res = as_path(d)
        return res.append(self)

    def append(self, d) -> Path:
        return Path._resolve(Path._parser_iter(d), self)

    def extend(self, d: list) -> Path:
        return Path._resolve(d, self)

    def with_suffix(self, pth: str) -> Path:
        pth = Path(pth)
        if len(self) == 0:
            return pth
        else:
            res = copy(self)
            if isinstance(res[-1], str) and isinstance(pth[0], str):
                res[-1] += pth[0]
                res.extend(pth[1:])
            else:
                res.extend(pth[:])
        return res

    def __truediv__(self, p) -> Path:
        return copy(self).append(p)

    def __add__(self, p) -> Path:
        return copy(self).append(p)

    def __iadd__(self, p) -> Path:
        return self.append(p)

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
            return Path(self[:idx] + self[idx + 1 :])

    @staticmethod
    def reduce(path: list) -> list:
        if len(path) < 2:
            return path
        elif isinstance(path[0], set) and path[1] in path[0]:
            return Path.reduce(path[1:])
        elif isinstance(path[0], slice) and isinstance(path[1], int):
            start = path[0].start if path[0].start is not None else 0
            step = path[0].step if path[0].step is not None else 1
            stop = start + step * path[1]
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
            res = Path._parser_str(p)
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

    @staticmethod
    def _resolve(source, target: list | None = None) -> list:
        """Make the path absolute, resolving all Path.tags (i.e. tag.root, tags.parent)"""
        if target is None:
            target = []

        for p in source:
            if p is Path.tags.parent:
                if len(target) > 0:
                    target.pop()
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
        elif isinstance(p, int):
            return str(p)
        elif isinstance(p, tuple):
            m_str = ",".join([Path._to_str(s) for s in p])
            return f"({m_str})"
        elif isinstance(p, set):
            m_str = ",".join([Path._to_str(s) for s in p])
            return f"{{{m_str}}}"
        elif isinstance(p, slice):
            if p.start is None and p.stop is None and p.step is None:
                return "*"
            else:
                return f"{p.start}:{p.stop}:{p.step}"
        elif p is None:
            return ""
        else:
            return str(p)
            # raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    # example:
    # a/b_c6/c[{value:{$le:10}}][value]/D/[1，2/3，4，5]/6/7.9.8

    PATH_PATTERN = re.compile(r"(?P<key>[^\[\]\/\,]+)(\[(?P<selector>[^\[\]]+)\])?")

    # 正则表达式解析，匹配一段被 {} 包裹的字符串
    PATH_REGEX_DICT = re.compile(r"\{(?P<selector>[^\{\}]+)\}")

    @staticmethod
    def _parser_selector(s: str | list) -> PathLike:
        if isinstance(s, str):
            s = s.strip(" ")

        if not isinstance(s, str):
            item = s
        elif s.startswith(("[", "(", "{")) and s.endswith(("}", ")", "]")):
            tmp = ast.literal_eval(s)
            if isinstance(tmp, dict):
                item = Query(tmp)  # {Path._parser_str_one(k): d for k, d in tmp.items()}
            elif isinstance(tmp, set):
                item = set([Path._parser_selector(k) for k in tmp])
            elif isinstance(tmp, tuple):
                item = tuple([Path._parser_selector(k) for k in tmp])
            elif isinstance(tmp, list):
                item = [Path._parser_selector(k) for k in tmp]

        elif s.startswith("(") and s.endswith(")"):
            tmp: dict = ast.literal_eval(s)
            item = {Path._parser_selector(k): d for k, d in tmp.items()}
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
        elif s == "...":
            item = Path.tags.ancestors
        elif s == "**":
            item = Path.tags.descendants
        elif s == ".":
            item = Path.tags.current
        elif s.isnumeric():
            item = int(s)
        elif s.startswith("$") and hasattr(Path.tags, s[1:]):
            item = Path.tags[s[1:]]
        else:
            item = s

        return item

    @staticmethod
    def _parser_iter(path: typing.Any) -> typing.Generator[PathLike, None, None]:
        if not isinstance(path, str):
            if isinstance(path, collections.abc.Iterable):
                for item in path:
                    if isinstance(item, str):
                        yield from Path._parser_iter(item)
                    else:
                        yield item
            else:
                yield path

        else:
            if path.startswith("/"):
                yield Path.tags.root
                path = path[1:]
            elif path.isidentifier():
                yield path
                return

            for match in Path.PATH_PATTERN.finditer(path):
                key = match.group("key")

                if key is None:
                    pass

                elif (tmp := is_int(key)) is not False:
                    yield tmp

                elif key == "*":
                    yield Path.tags.children

                elif key == "..":
                    yield Path.tags.parent

                elif key == "...":
                    yield Path.tags.ancestors

                else:
                    yield key

                selector = match.group("selector")
                if selector is not None:
                    yield Path._parser_selector(selector)

    @staticmethod
    def _parser(path) -> list:
        """Parse the PathLike to list"""
        return [*Path._parser_iter(path)]

    ###########################################################
    # RESTful API:

    # 非幂等
    def insert(self, target: typing.Any, *args, **kwargs) -> typing.Tuple[typing.Any, Path]:
        """
        根据路径（self）向 target 添加元素。
        当路径指向位置为空时，创建（create）元素
        当路径指向位置为 list 时，追加（ insert ）元素
        当路径指向位置为非 list 时，合并为 [old,new]
        当路径指向位置为 dict, 添加值亦为 dict 时，根据 key 递归执行 insert

        返回新添加元素的路径

        对应 RESTful 中的 post，非幂等操作
        """
        return Path._do_update(target, self[:], *args, _idempotent=False, **kwargs)

    # 幂等
    def update(self, target: typing.Any, *args, **kwargs) -> typing.Any:
        """
        根据路径（self）更新 target 中的元素。
        当路径指向位置为空时，创建（create）元素
        当路径指向位置为 dict, 添加值亦为 dict 时，根据 key 递归执行 update
        当路径指向位置为空时，用新的值替代（replace）元素

        对应 RESTful 中的 put， 幂等操作

        返回修改后的target
        """
        return Path._do_update(target, self[:], *args, **kwargs)

    def remove(self, target: typing.Any, *args, **kwargs) -> typing.Tuple[typing.Any, int]:
        """根据路径（self）删除 target 中的元素。

        if quiet is False then raise KeyError if the path is not found

        对应 RESTful 中的 delete， 幂等操作

        返回修改后的target和删除的元素的个数
        """
        return Path._do_update(target, self[:], None, *args, **kwargs)

    def find(self, source: typing.Any, *args, **kwargs) -> typing.Any:
        """
        根据路径（self）查询元素。只读，不会修改 target
        对应 RESTful 中的 read，幂等操作
        """
        return Path._do_find(source, self[:], *args, **kwargs)

    def for_each(self, source, *args, **kwargs) -> typing.Generator[typing.Tuple[int | str, typing.Any], None, None]:
        yield from Path._do_for_each(source, self[:], *args, **kwargs)

    def search(self, source, query, *args, **kwargs) -> int | None:
        return Path._do_find(source, self[:], Path.tags.search, query, *args, **kwargs)

    def keys(self, source) -> typing.Generator[str, None, None]:
        yield from Path._do_for_each(source, Path.tags.keys)

    def copy(self, source: typing.Any, target: typing.Any = None, *args, **kwargs):
        """copy object to target"""
        return self._do_copy(source, target, self[:], *args, **kwargs)

    # --------------------------------------------------------------------------------------
    # 以下为简写，等价于上述方法，不应被重载

    def put(self, target: typing.Any, value) -> typing.Any:
        return self.update(target, value)

    def get(self, source: typing.Any, default_value=_not_found_):
        return self.find(source, default_value=default_value)

    # End API
    ###########################################################

    @staticmethod
    def _apply_op(obj: typing.Any, op: Path.tags, *args, **kwargs):
        if op is Path.tags.find:
            return obj

        elif isinstance(op, Path.tags):
            func = getattr(Path, f"_op_{op.name}", None)
        else:
            func = op

        if not callable(func):
            raise RuntimeError(f"Unknown operator {op}")

        try:
            res = func(obj, *args, **kwargs)
        except Exception as error:
            raise RuntimeError(f'Fail to call "{op}"! {func}') from error

        return res

    @staticmethod
    def _do_copy(source: typing.Any, target: typing.Any, *args, **kwargs) -> typing.Any:
        raise NotImplementedError(f"TODO")

    @staticmethod
    def _update_tree(target, *sources, _idempotent=True, **kwargs):
        if len(kwargs) > 0:
            sources = [*sources, kwargs]

        for source in sources:
            if source is _not_found_:  # 不更新
                continue
            elif (target is _not_found_ or target is None) and not isinstance(source, dict):  # 直接替换
                target = source
            elif isinstance(source, dict):  # 合并 dict
                # if not isinstance(target, dict) and hasattr(target.__class__, "update"):
                #     # target is object with update method
                #     target.update(source)
                # else:
                if target is _not_found_ or target is None:
                    target = {}

                for key, value in source.items():
                    Path(key).update(target, value, _idempotent=_idempotent)

            elif _idempotent is True:  # 幂等操作，直接替换对象 target
                target = source

            elif isinstance(target, list):  # 合并 sequence
                if isinstance(source, list):
                    target.extend(source)
                else:
                    target.append(source)
            elif isinstance(source, list):
                target = [target, *source]
            else:
                target = [target, source]

        return target

    @staticmethod
    def _do_update(target: typing.Any, path: list, *args, **kwargs) -> typing.Any:
        if path is None or len(path) == 0:
            if len(args) > 0 and isinstance(args[0], Path.tags):
                target = Path._apply_op(target, *args, **kwargs)

            else:
                idempotent = kwargs.get("_idempotent", True)
                extend = kwargs.get("_extend", False)

                for value in args:
                    if value is _not_found_:  # 空操作
                        continue

                    elif value is target:
                        continue

                    elif (value is None and idempotent) or target is _not_found_ or target is None:  # 删除或取代
                        target = value

                    elif isinstance(target, collections.abc.MutableMapping):
                        if isinstance(value, dict):
                            for k, v in value.items():
                                target = Path._do_update(target, as_path(k)[:], v, **kwargs)
                        else:
                            target = value  # 取代

                    elif idempotent:
                        target = value

                    elif isinstance(target, collections.abc.MutableSequence):
                        if extend and isinstance(value, (list)):
                            target.extend(value)
                        else:
                            target.append(value)
                    elif extend and isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                        target = [target, *value]
                    else:
                        target = [target, value]

        else:
            key = path[0]

            if target is _not_found_ or target is None:
                target = {} if isinstance(key, str) else []

            if key is None:
                target = Path._do_update(target, path[1:], *args, **kwargs)

            elif isinstance(key, str) and key.isidentifier() and hasattr(target.__class__, key):
                if len(path) == 1:
                    value = Path._do_update(_not_found_, [], *args, **kwargs)
                    setattr(target, key, value)

                else:
                    attr = getattr(target, key, _not_found_)
                    value = Path._do_update(attr, path[1:], *args, **kwargs)

                    if value is not attr:
                        setattr(target, key, value)

            elif isinstance(target, collections.abc.MutableMapping):
                target[key] = Path._do_update(target.get(key, _not_found_), path[1:], *args, **kwargs)

            elif isinstance(target, collections.abc.MutableSequence):
                if isinstance(key, int):
                    require_length = key

                elif isinstance(key, slice):
                    require_length = slice.stop

                else:
                    require_length = None

                if require_length is not None and require_length >= (len(target)):
                    target.extend([_not_found_] * (key - len(target) + 1))

                if isinstance(key, int):
                    target[key] = Path._do_update(target[key], path[1:], *args, **kwargs)

                elif isinstance(key, slice):
                    target[key] = [Path._do_update(obj, path[1:], *args, **kwargs) for obj in target[key]]

                else:
                    query = Query(key)

                    for idx, d in enumerate(target):
                        if query.check(d):
                            target[idx] = Path._do_update(d, path[1:], *args, **kwargs)
                            break
                    else:
                        target.append({f"@{Path.id_tag_name}": key})
                        target = Path._do_update(target, path, *args, **kwargs)

            else:
                raise TypeError(f"Can not update {type(target)} '{target}' path={path} {args} {kwargs}")

        return target

    @staticmethod
    def _do_find(source: typing.Any, path: list, *args, **kwargs) -> typing.Any:
        if not isinstance(path, list):
            raise TypeError(path)

        if source is _not_found_ or source is None:
            res = source

        elif path is not None and len(path) > 0:
            key = path[0]

            is_attr = False

            if (
                isinstance(key, str)
                and key.isidentifier()
                and (attr := getattr(source, key, _not_found_)) is not _not_found_
            ):
                is_attr = True
                res = Path._do_find(attr, path[1:], *args, **kwargs)

            if is_attr:
                pass
            elif key is None:
                res = Path._do_find(source, path[1:], *args, **kwargs)

            elif source is _not_found_ or source is None:
                res = source

            elif len(path) > 1 and path[1] is Path.tags.next:
                if isinstance(key, int):
                    new_key = key + 1
                else:
                    raise NotImplementedError(f"{type(source)} {path}")

                res = Path._do_find(source, [new_key] + path[2:], *args, **kwargs)

            elif len(path) > 1 and path[1] is Path.tags.prev:
                if isinstance(key, int):
                    new_key = key - 1
                else:
                    raise NotImplementedError(f"{type(source)} {path}")

                res = Path._do_find(source, [new_key] + path[2:], *args, **kwargs)

            elif key is Path.tags.current:
                res = Path._do_find(source, path[1:], *args, **kwargs)

            elif key is Path.tags.parent:
                res = Path._do_find(getattr(source, "_parent", _not_found_), path[1:], *args, **kwargs)

            elif key is Path.tags.ancestors:
                # 逐级查找上层 _parent, 直到找到
                obj = source

                default_value = kwargs.pop("default_value", _not_found_)

                while obj is not None and obj is not _not_found_:
                    if not isinstance(obj, collections.abc.Sequence):
                        res = Path._do_find(obj, path[1:], *args, default_value=_not_found_, **kwargs)
                        if res is not _not_found_:
                            break
                    obj = getattr(obj, "_parent", _not_found_)
                    # Path._do_find(obj, [Path.tags.parent], default_value=_not_found_)
                else:
                    res = default_value

            elif key is Path.tags.descendants:
                # 遍历访问所有叶节点
                if isinstance(source, collections.abc.Mapping):
                    res = {
                        k: Path._do_find(v, [Path.tags.descendants] + path[1:], *args, **kwargs)
                        for k, v in source.items()
                    }

                elif isinstance(source, collections.abc.Iterable):
                    res = [Path._do_find(v, [Path.tags.descendants] + path[1:], *args, **kwargs) for v in source]

                else:
                    res = Path._do_find(source, path[1:], *args, **kwargs)

            elif key is Path.tags.children:
                if isinstance(source, collections.abc.Mapping):
                    res = {k: Path._do_find(v, path[1:], *args, **kwargs) for k, v in source.items()}

                elif isinstance(source, collections.abc.Iterable):
                    res = [Path._do_find(v, path[1:], *args, **kwargs) for v in source]

                else:
                    res = []

            elif key is Path.tags.slibings:
                parent = Path._do_find(source, [Path.tags.parent], default_value=_not_found_)

                if isinstance(parent, collections.abc.Mapping):
                    res = {k: Path._do_find(v, path[1:], *args, **kwargs) for k, v in parent.items() if v is not source}

                elif isinstance(parent, collections.abc.Iterable):
                    res = [Path._do_find(v, path[1:], *args, **kwargs) for v in parent if v is not source]

                else:
                    res = []

            elif isinstance(source, collections.abc.Mapping):
                res = Path._do_find(source.get(key, _not_found_), path[1:], *args, **kwargs)

            elif isinstance(source, collections.abc.Sequence) and not isinstance(source, str):
                if isinstance(key, int):
                    res = Path._do_find(source[key], path[1:], *args, **kwargs)

                elif isinstance(key, slice):
                    res = [Path._do_find(s, path[1:], *args, **kwargs) for s in source[key]]

                else:
                    idx, value = Path._op_search(source, key)

                    if idx is None:
                        res = _not_found_
                    else:
                        res = Path._do_find(value, path[1:], *args, **kwargs)

                    # if not isinstance(source, list) and hasattr(source.__class__, "__getitem__"):
                    #     # 先尝试默认的 __getitem__
                    #     try:
                    #         res = source[key]
                    #     except Exception as error:
                    #         (error)
                    #         res = _not_found_
                    #     else:
                    #         res = Path._do_find(res, path[1:], *args, **kwargs)

                    # if res is _not_found_:
                    #     if isinstance(key, str) and not key.startswith(("@", "$")):
                    #         query = Query({f"@{Path.id_tag_name}": key})
                    #     else:
                    #         query = Query(key)

                    #     for d in source:
                    #         if query.check(d):
                    #             res = Path._do_find(d, path[1:], *args, **kwargs)
                    #             break
                    #     else:
                    #         res = _not_found_

            else:
                res = _not_found_

            if res is _not_found_ and len(args) > 0 and (isinstance(args[0], Path.tags) or callable(args[0])):
                res = Path._apply_op(_not_found_, *args, **kwargs)

        elif len(args) == 0 or args[0] is None:
            res = source

        elif isinstance(args[0], Path.tags) or (isinstance(args[0], Path.tags) or callable(args[0])):
            res = Path._apply_op(source, *args, **kwargs)

        else:
            res = source
            if len(args) > 0:
                raise RuntimeError(f"ignore {args} {kwargs}")

        if res is _not_found_:
            res = kwargs.get("default_value", _not_found_)

        return res

    @staticmethod
    def _do_for_each(source, path, *args, level=0, **kwargs) -> typing.Generator[typing.Any, None, None]:
        if level < 0:
            yield None, source
        elif path is None or len(path) == 0:
            if isinstance(source, collections.abc.Sequence) and not isinstance(source, str):
                yield from enumerate(source)
            elif isinstance(source, collections.abc.Mapping):
                yield from source.items()
            else:
                yield None, source
        else:
            key = path[0]

            if key is None:
                yield from Path._do_for_each(source, path[1:], *args, level=level, **kwargs)

            elif key is Path.tags.children:
                if isinstance(source, collections.abc.Mapping):
                    for k, v in source.items():
                        yield (k, Path._do_for_each(v, path[1:], *args, level=level - 1, **kwargs))

                elif isinstance(source, collections.abc.Iterable):
                    for idx, v in enumerate(source):
                        yield idx, Path._do_for_each(v, path[1:], *args, level=level - 1, **kwargs)

            elif key is Path.tags.slibings:
                parent = Path._do_find(source, [Path.tags.parent], default_value=_not_found_)

                if isinstance(parent, collections.abc.Mapping):
                    for k, v in parent.items():
                        if v is source:
                            continue
                        yield k, Path._do_for_each(v, path[1:], *args, level=level - 1, **kwargs)

                elif isinstance(parent, collections.abc.Iterable):
                    for idx, v in enumerate(parent):
                        if v is source:
                            continue
                        yield idx, Path._do_for_each(v, path[1:], *args, level=level - 1, **kwargs)

            elif key is Path.tags.ancestors:
                key = ""
                while source is not None and source is not _not_found_:
                    res = Path._do_find(source, path[1:], *args, **kwargs)
                    if res is not _not_found_:
                        yield key + "/".join(path[1:]), res
                        break
                    source = getattr(source, "_parent", _not_found_)
                    key += "../"

            elif key is Path.tags.descendants:
                if isinstance(source, collections.abc.Mapping):
                    for i, v in source.items():
                        for j, s in Path._do_for_each(v, [Path.tags.descendants] + path[1:], *args, **kwargs):
                            yield f"{i}/{j}", s
                elif isinstance(source, collections.abc.Iterable):
                    for i, v in enumerate(source):
                        for j, s in Path._do_for_each(v, [Path.tags.descendants] + path[1:], *args, **kwargs):
                            yield f"{i}/{j}", s
                else:
                    obj = Path._do_find(source, [key], default_value=_not_found_)
                    yield from Path._do_for_each(obj, path[1:], *args, **kwargs)

            elif isinstance(key, Query):
                raise NotImplementedError(f"Not implemented! {path}")

            else:
                source = Path._do_find(source, [key], default_value=_not_found_)

                for p, v in Path._do_for_each(source, path[1:], *args, level=level, **kwargs):
                    yield p, v
                    if p is None:
                        break

    ####################################################
    # operation

    @staticmethod
    def _op_is_leaf(source: typing.Any, *args, **kwargs) -> bool:
        return not isinstance(source, (collections.abc.Mapping, collections.abc.Sequence))

    @staticmethod
    def _op_is_list(source: typing.Any, *args, **kwargs) -> bool:
        return isinstance(source, collections.abc.Sequence)

    @staticmethod
    def _op_is_dict(source: typing.Any, *args, **kwargs) -> bool:
        return isinstance(source, collections.abc.Mapping)

    @staticmethod
    def _op_check_type(source: typing.Any, tp, *args, **kwargs) -> bool:
        return isinstance_generic(source, tp)

    @staticmethod
    def _op_count(source: typing.Any, *args, **kwargs) -> int:
        if source is _not_found_:
            return 0
        elif isinstance(source, collections.abc.Sequence) or isinstance(source, collections.abc.Mapping):
            return len(source)
        else:
            return 1

    @staticmethod
    def _op_fetch(source: typing.Any, *args, default_value=_not_found_, **kwargs) -> bool:
        if source is _not_found_:
            source = default_value
        return source

    @staticmethod
    def _op_exists(source: typing.Any, *args, **kwargs) -> bool:
        return source is not _not_found_

    @staticmethod
    def _op_search(source: typing.Iterable, *args, search_range=None, **kwargs):
        if not isinstance(source, collections.abc.Sequence):
            raise TypeError(f"{type(source)} is not sequence")

        if len(args) == 1 and isinstance(args[0], str):
            query = Query({f"@{Path.id_tag_name}": args[0], **kwargs})
        else:
            query = Query(*args, **kwargs)

        if search_range is not None and isinstance(source, collections.abc.Sequence):
            source = source[search_range]

        for idx, value in enumerate(source):
            if query.check(value):
                break
        else:
            idx = None
            value = _not_found_

        return idx, value

    @staticmethod
    def _op_call(source, *args, **kwargs) -> typing.Any:
        if not callable(source):
            logger.error(f"Not callable! {source}")
            return _not_found_
        else:
            return source(*args, **kwargs)

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


PathLike = str | int | slice | dict | list | OpTags | Path | None

path_like = [int, str, slice, list, None, tuple, set, dict, OpTags, Path]


_T = typing.TypeVar("_T")


def update_tree(target: _T, *args, **kwargs) -> _T:
    return Path().update(target, *args, **kwargs)


def merge_tree(*args, **kwargs) -> _T:
    return update_tree({}, *args, **kwargs)


def as_path(path):
    if path is None or path is _not_found_:
        return Path()
    elif not isinstance(path, Path):
        return Path(path)
    else:
        return path
