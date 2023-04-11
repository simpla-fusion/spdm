from __future__ import annotations

import ast
import collections.abc
import pprint
import re
import typing
from copy import deepcopy
from enum import Flag, auto

import numpy as np

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger


class Path(list):
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

    DELIMITER = '/'
    _PRIMARY_INDEX_TYPE_ = (int, float)

    class tags(Flag):
        # traversal operation 操作

        next = auto()
        parent = auto()
        root = auto()

        append = auto()
        extend = auto()
        update = auto()
        sort = auto()
        deep_update = auto()
        setdefault = auto()
        # predicate 谓词
        count = auto()
        equal = auto()
        le = auto()
        ge = auto()
        less = auto()
        greater = auto()

    @classmethod
    def normalize(cls, p: typing.Any, raw=False) -> typing.Any:
        if p is None:
            res = []
        elif isinstance(p, Path):
            res = p[:]
        elif isinstance(p, str):
            res = cls._parser(p)
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

    @classmethod
    def _to_str(cls, p: typing.Any) -> str:
        if isinstance(p, str):
            return p
        elif isinstance(p, slice):
            return f"{p.start}:{p.stop}:{p.step}"
        elif isinstance(p, int):
            return str(p)
        elif isinstance(p, collections.abc.Mapping):
            m_str = ','.join([f"{k}:{Path._to_str(v)}" for k, v in p.items()])
            return f"?{{{m_str}}}"
        elif isinstance(p, list):
            return '/'.join(map(Path._to_str, p))
        elif isinstance(p, tuple):
            m_str = ','.join(map(Path._to_str, p))
            return f"({m_str})"
        elif isinstance(p, set):
            m_str = ','.join(map(Path._to_str, p))
            return f"{{{m_str}}}"
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    # example:
    # a/b_c6/c[{value:{$le:10}}][value]/D/[1，2/3，4，5]/6/7.9.8
    PATH_REGEX = re.compile(r"(?P<key>[^\[\]\/\,\.]+)|(\[(?P<selector>[^\[\]]+)\])")

    # 正则表达式解析，匹配一段被 {} 包裹的字符串
    PATH_REGEX_DICT = re.compile(r"\{(?P<selector>[^\{\}]+)\}")

    @classmethod
    def _parser(cls, v: str) -> list:
        """
        """

        def parser_one(v: str) -> typing.Any:
            if v.startswith(("[", "(", "{")) and v.endswith(("]", "}", ")")):
                try:
                    res = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    try:
                        res = slice(*map(int, v[1:-1].split(':')))
                    except ValueError:
                        raise ValueError(f"Invalid Path: {v}")
            else:
                if v.isnumeric():
                    res = int(v)
                elif v.startswith("$"):
                    res = Path.tags[v[1:]]
                else:
                    res = v
            return res
        res = [*map(parser_one, v.split(Path.DELIMITER))]
        if len(res) == 1:
            res = res[0]
        return res

    @classmethod
    def parser(cls, p: str) -> Path:
        return Path(Path._parser(p))

    def __init__(self, *args, **kwargs):
        super().__init__(Path.normalize(list(args)), **kwargs)

    def __repr__(self):
        return pprint.pformat(self)

    def __str__(self):
        return Path._to_str(self)

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def duplicate(self, new_value=None) -> Path:
        if new_value is not None:
            return self.__class__(new_value)
        else:
            return self.__class__(self[:])

    def as_list(self) -> list:
        return self[:]

    def as_url(self) -> str:
        return '/'.join(map(Path._to_str, self))

    @property
    def is_closed(self) -> bool:
        return len(self) > 0 and self[-1] is None

    @property
    def is_leaf(self) -> bool:
        return len(self) > 0 and self[-1] is None

    @property
    def is_root(self) -> bool:
        return len(self) == 0

    @property
    def is_regular(self) -> bool:
        return next((i for i, v in enumerate(self[:]) if not isinstance(v, Path._PRIMARY_INDEX_TYPE_)), None) is None

    def close(self) -> Path:
        if not self.is_closed:
            self.append(None)
        return self

    def open(self) -> Path:
        if self[-1] is None:
            self.pop()
        return self

    @property
    def parent(self) -> Path:
        if self.is_root:
            raise RuntimeError("Root node hasn't parents")
        other = self.duplicate()
        other.pop()
        return other

    @property
    def children(self) -> Path:
        if self.is_leaf:
            raise RuntimeError("Leaf node hasn't child!")
        other: Path = self.duplicate()
        other.append(slice(None))
        return other

    @property
    def slibings(self):
        return self.parent.children

    @property
    def next(self) -> Path:
        other = self.duplicate()
        other.append(Path.tags.next)
        return other

    def append(self, *args, force=False) -> Path:
        if self.is_closed:
            raise ValueError(f"Cannot append to a closed path {self}")
        if force:
            super().extend(list(args))
        else:
            super().extend(Path.normalize(list(args)))
        return self

    def __truediv__(self, p) -> Path:
        return self.duplicate().append(p)

    def __add__(self, p) -> Path:
        return self.duplicate().append(p)

    def __iadd__(self, p) -> Path:
        return self.append(p)

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
        yield from self._find(target, self[:], *args, **kwargs)

    def query(self, target: typing.Any, *args, **kwargs) -> typing.Any:
        return self._query(target, self[:], *args, **kwargs)

    def insert(self, target: typing.Any, *args, **kwargs) -> int:
        return self._insert(target, self[:], *args, **kwargs)

    def remove(self, target: typing.Any, *args, **kwargs) -> int:
        return self._remove(target, self[:], *args, **kwargs)

    def update(self, target: typing.Any, *args, **kwargs) -> int:
        return self._update(target, self[:], *args,  **kwargs)

    # End API

    def _traversal(self, target: typing.Any, path: typing.List[typing.Any]) -> typing.Tuple[typing.Any, int]:
        """
        Traversal the target with the path, return the last regular target and the position the first non-regular path.
        :param target: the target to traversal
        :param path: the path to traversal

        """
        pos = -1
        for idx, p in enumerate(path):
            if hasattr(target, "__entry__"):
                break
            elif isinstance(p, str):
                if not isinstance(target, collections.abc.Mapping):
                    raise TypeError(f"Cannot traversal {p} in {type(target)}")
                elif p not in target:
                    break
            elif isinstance(p, int):
                if isinstance(target, np.ndarray):
                    target = target[p]
                elif not isinstance(target, (list, tuple, collections.deque)):
                    raise TypeError(f"Cannot traversal {p} in {type(target)} {target}")
                elif p >= len(target):
                    raise IndexError(f"Index {p} out of range {len(target)}")
            elif isinstance(p, tuple) and all(isinstance(v, (int, slice)) for v in p):
                if not isinstance(target, (np.ndarray)):
                    break
            else:
                break
            target = target[p]
            pos = idx
        return target, pos+1

    def _find(self, target: typing.Any, path: typing.List[typing.Any], *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        target, pos = self._traversal(target, path)

        def concate_path(p, others: list):
            if not isinstance(p, list):
                return [p]+others
            else:
                return p+others
        if len(path) == pos:
            yield target  # target is the last node
        elif hasattr(target, "__entry__"):
            yield target.__entry__.child(path[pos:]).find(*args, **kwargs)
        elif isinstance(path[pos], str):
            if "default_value" in kwargs:
                yield target.get(path[pos], kwargs["default_value"])
            else:
                raise TypeError(f"{type(path[pos])}")
        elif isinstance(path[pos], set):
            yield from ((k, self._find(target, concate_path(k, path[pos+1:]),  *args, **kwargs)) for k in path[pos])
        elif isinstance(path[pos], tuple):
            yield from (self._find(target, concate_path(k, path[pos+1:]),  *args, **kwargs) for k in path[pos])
        elif isinstance(path[pos], slice):
            if isinstance(target, (np.ndarray)):
                yield from self._find(target[path[pos]], path[pos+1:],  *args, **kwargs)
            elif isinstance(target, (collections.abc.Sequence)) and not isinstance(target, str):
                for item in target[path[pos]]:
                    yield from self._find(item, path[pos+1:],  *args, **kwargs)
            elif "default_value" in kwargs:
                yield kwargs["default_value"]
            else:
                raise TypeError(f"Cannot slice {type(target)}")
        elif isinstance(path[pos], collections.abc.Mapping):
            only_first = kwargs.get("only_first", False)
            if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
                for element in target:
                    if self._match(element, path[pos]):
                        yield from self._find(element,  path[pos+1:],  *args, **kwargs)
                        if only_first:
                            break
            elif "default_value" in kwargs:
                yield kwargs["default_value"]
            else:
                raise TypeError(f"Cannot search {type(target)}")
        elif "default_value" in kwargs:
            yield kwargs["default_value"]
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {path[pos]}")

    def _match(self, target: typing.Any, predicate: typing.Mapping[str, typing.Any]) -> bool:
        """

        """
        if not isinstance(predicate, collections.abc.Mapping):
            predicate = {predicate: None}

        def do_match(op, value, expected):
            res = False
            if isinstance(op, Path.tags):
                res = Path._ops[op](value, expected)
            else:
                try:
                    actual, p = Entry._eval_path(value, Entry.normalize_path(op)+[None])
                    res = p is None and (actual == expected)
                except (IndexError, KeyError):
                    res = False

            return res

        return all([do_match(op, target, args) for op, args in predicate.items()])

    def _query(self, target: typing.Any, path: typing.List[typing.Any], op: typing.Optional[Path.tags] = None, *args, **kwargs) -> typing.Any:
        target, pos = self._traversal(target, path)

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).query(op, *args, **kwargs)
        elif pos < len(path) - 1 and not isinstance(path[pos], (int, str)):
            return [self._query(d, path[pos+1:], op, *args, **kwargs) for d in self._find(target, path[pos], **kwargs)]
        elif pos == len(path) - 1 and not isinstance(path[pos], (int, str)):
            target = self._expand(self._find(target, path[pos:], **kwargs))
            return self._query(target, [], op, *args, **kwargs)
        elif op is None:
            if pos < len(path):  # Not found
                default_value = kwargs.get("default_value", _undefined_)
                if default_value is _undefined_:
                    raise KeyError(f"Cannot find {path[:pos]} in {target}")
                else:
                    return default_value
            else:
                return self._expand(self._find(target, path[pos:], **kwargs))
        elif op == Path.tags.count:
            if pos == len(path):
                if isinstance(target, str):
                    return 1
                else:
                    return len(target)
            else:
                return 0
        elif op == Path.tags.equal:
            if pos == len(path):
                return target == args[0]
            else:
                return False
        else:
            raise NotImplementedError(f"Not implemented yet! {op}")

    def _insert(self, target: typing.Any, path: typing.List[typing.Any], value: typing.Any, *args, parents=True, **kwargs) -> int:
        target, pos = self._traversal(target, path[:-1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).insert(value, *args, parents=parents, **kwargs)
        elif len(path) == 0:
            return self._update(target, value, *args, **kwargs)
        elif pos < len(path)-1 and not isinstance(path[pos], (int, str)):
            return sum(self._insert(d, path[pos+1:], value, *args, **kwargs) for d in self._find(target, [path[pos]], **kwargs))
        elif not parents:
            raise IndexError(f"Can't insert {value} to {target} by {path[:pos]}!")
        else:
            for p in path[pos:-1]:
                target = target.setdefault(p, {})
            target[path[-1]] = value
            return 1

    def _remove(self, target: typing.Any, path: typing.List[typing.Any],  *args, **kwargs) -> int:
        """
        Remove target by path.
        """

        target, pos = self._traversal(target, path[:-1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).delete(*args, **kwargs)
        elif len(path) == 0:
            target.clear()
            return 1
        elif pos < len(path)-1 and not isinstance(path[pos], (int, str)):
            return sum(self._remove(d, path[pos+1:], *args, **kwargs) for d in self._find(target, [path[pos]], **kwargs))
        elif pos < len(path)-1:
            return 0
        else:
            del target[path[-1]]
            return 1

    def _update_or_replace(self, target: typing.Any, actions: collections.abc.Mapping,
                           force=False, replace=True, **kwargs) -> typing.Any:
        if not isinstance(actions, collections.abc.Mapping):
            return actions

        for op, args in actions.items():
            if isinstance(op, str) and op.startswith("$"):
                op = Path.tags[op[1:]]

            if isinstance(op, str):
                if target is None:
                    target = {}
                self._update(target, [op], args, force=force, **kwargs)
            elif op in (Path.tags.append,  Path.tags.extend):
                new_obj = self._update_or_replace(None, args, force=True, replace=True)
                if op is Path.tags.append:
                    new_obj = [new_obj]

                if isinstance(target, list):
                    target += new_obj
                elif replace:
                    target = [target] + new_obj
                else:
                    raise IndexError(f"Can't append {new_obj} to {target}!")
            else:
                raise NotImplementedError(f"Not implemented yet!{op}")

        return target

    def _update(self, target: typing.Any, path: typing.List[typing.Any], value:  typing.Any, *args,  force=False, **kwargs) -> int:
        """
        Update target by path with value.
        """
        target, pos = self._traversal(target, path[:-1])

        if hasattr(target, "__entry__"):
            return target.__entry__.child(path[pos:]).update(value,  *args, overwrite=force, **kwargs)
        elif len(path) == 0:
            self._update_or_replace(target, value, force=force, replace=False, **kwargs)
            return 1
        elif not isinstance(path[pos], (int, str)):
            return sum(self._update(d, path[pos+1:], value, *args,  force=force, **kwargs)
                       for d in self._find(target, [path[pos]], **kwargs))
        elif not isinstance(value, collections.abc.Mapping):
            return self._insert(target, path[pos:], value, *args, force=force, **kwargs)
        elif pos < len(path)-1:
            return self._insert(target, path[pos:], self._update_or_replace(None, value), force=force, **kwargs)
        else:  # pos == len(path)-1
            n_target, n_pos = self._traversal(target, path[pos:])
            if n_pos == 0:
                return self._update(target, [], value, force=force, **kwargs)
            else:
                return self._insert(target, path[pos:], self._update_or_replace(n_target, value), force=force, **kwargs)

    def _expand(self, target: typing.Any):
        if isinstance(target, collections.abc.Generator):
            res = [self._expand(v) for v in target]
            if len(res) > 1 and isinstance(res[0], tuple) and len(res[0]) == 2:
                res = dict(*res)
            elif len(res) == 1:
                res = res[0]
        else:
            res = target

        return res


class obsolete_Path:

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

    def _op_find(self, k, default_value=_undefined_):
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

    def _op_by_filter(self, pred, op,  *args, on_fail: Callable = _undefined_):
        if not isinstance(self, collections.abc.Sequence):
            raise TypeError(type(self))

        if isinstance(pred, collections.abc.Mapping):
            def pred(val, _cond=pred):
                if not isinstance(val, collections.abc.Mapping):
                    return False
                else:
                    return all([val.get(k, _not_found_) == v for k, v in _cond.items()])

        res = [op(self, idx, *args)
               for idx, val in enumerate(self) if pred(val)]

        if len(res) == 1:
            res = res[0]
        elif len(res) == 0 and on_fail is not _undefined_:
            res = on_fail(self)
        return res

    def _op_assign(self, path, v):
        self, key = Entry._eval_path(
            self,  Entry.normalize_path(path), force=True, lazy=False)
        if not isinstance(key, (int, str, slice)):
            raise KeyError(path)
        elif not isinstance(self, (collections.abc.Mapping, collections.abc.Sequence)):
            raise TypeError(type(self))
        self[key] = v
        return v

    def _op_insert(self, k, v):
        if isinstance(self, collections.abc.Mapping):
            val = self.get(k, _not_found_)
        else:
            val = self[k]

        if val is _not_found_:
            self[k] = v
            val = v

        return val

    def _op_append(self, k,  v):
        if isinstance(self, Entry):
            self = self.get(k,  _LIST_TYPE_())
        else:
            self = self.setdefault(k, _LIST_TYPE_())

        if not isinstance(self, collections.abc.Sequence):
            raise TypeError(type(self))

        self.append(v)

        return v

    def _op_remove(self, k, *args):
        try:
            del self[k]
        except Exception as error:
            success = False
        else:
            success = True
        return success

    def _op_update(self, value):

        if not isinstance(self, collections.abc.Mapping):
            raise TypeError(type(self))
        elif value is None or value is _undefined_:
            return self

        for k, v in value.items():
            tmp = self.setdefault(k, v)

            if tmp is v or v is _undefined_:
                pass
            elif not isinstance(tmp, collections.abc.Mapping):
                self[k] = v
            elif isinstance(v, collections.abc.Mapping):
                Entry._op_update(tmp,  v)
            else:
                raise TypeError(type(v))

        return self

    def _op_try_insert(self, key, v):
        if isinstance(self, collections.abc.Mapping):
            val = self.setdefault(key, v)
        elif isinstance(self, collections.abc.Sequence):
            val = self[key]
            if val is None or val is _not_found_:
                self[key] = v
                val = v
        else:
            raise RuntimeError(type(self))
        return val

    def _op_check(self, pred=None, *args) -> bool:

        if isinstance(pred, Entry.op_tag):
            return Entry._ops[pred](self, *args)
        elif isinstance(pred, collections.abc.Mapping):
            return all([Entry._op_check(Entry._eval_path(self, Entry.normalize_path(k), _not_found_), v) for k, v in pred.items()])
        else:
            return self == pred

    def _op_exist(self, path, *args):
        if path in (None, _not_found_, _undefined_):
            return self not in (None, _not_found_, _undefined_)
        else:
            self, path = Entry._eval_path(
                self, Entry.normalize_path(path), force=False)
            if isinstance(path, str):
                return path in self
            elif isinstance(path, int):
                return path < len(self)
            else:
                return False

    def _op_equal(self, value):
        return self == value

    def _op_count(self, path):
        if path not in (None, _not_found_, _undefined_):
            self, path = Entry._eval_path(self, Entry.normalize_path(path), force=False)
            try:
                self = self[path]
            except Exception:
                return 0
        return len(self)

    # _ops = {
    #     op_tag.assign: _op_assign,
    #     op_tag.update: _op_update,
    #     op_tag.append: _op_append,
    #     op_tag.remove: _op_remove,
    #     op_tag.try_insert: _op_try_insert,

    #     # read
    #     op_tag.find: _op_find,
    #     op_tag.equal: lambda self, other: self == other,
    #     op_tag.count: lambda self, *args: len(self) if self not in (None, _not_found_, _undefined_) else 0,
    #     op_tag.exists: lambda self, *args: self not in (None, _not_found_, _undefined_),
    #     op_tag.dump: lambda self, *args: as_native(self),

    #     op_tag.next: None,
    #     op_tag.parent: None,
    #     op_tag.first_child: None,
    # }

    @ staticmethod
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

    @ staticmethod
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
            elif isinstance(self, np.ndarray) and isinstance(key, (int, slice)):
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

    @ staticmethod
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

    @ staticmethod
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

    @ staticmethod
    def _eval_push(self, path: list, value=_undefined_, *args):
        if isinstance(self, Entry):
            return self.push(path, value, *args)
        # elif isinstance(self, EntryContainer):
        #     return self.put(path,  value, *args)

        if path is _undefined_:
            path = []
        elif not isinstance(path, list):
            path = [path]
        if not isinstance(value, np.ndarray) and value is _undefined_:
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
            return self.duplicate().move_to(path)
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
            convert data in cache to python native type and np.ndarray
            [str, bool, float, int, np.ndarray, Sequence, Mapping]:
        """
        return as_native(self._cache, *args, **kwargs)


# 测试代码
