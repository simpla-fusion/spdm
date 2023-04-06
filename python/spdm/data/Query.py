from __future__ import annotations

import collections.abc
import typing
from copy import deepcopy
from enum import Flag, auto

import numpy as np

from ..common.tags import _not_found_, _undefined_
from ..util.dict_util import deep_merge_dict
from .Path import Path

_T = typing.TypeVar("_T")
_TObject = typing.TypeVar("_TObject")
_TPath = typing.TypeVar("_TPath", int,  slice, str,  typing.Sequence,  typing.Mapping)
_TStandardForm = typing.TypeVar("_TStandardForm", bool, int,  float, str, np.ndarray, list, dict)
_TKey = typing.TypeVar('_TKey', int, str)
_TIndex = typing.TypeVar('_TIndex', int, slice, str, typing.Sequence, typing.Mapping)


def _slice_to_range(s: slice, length: int) -> range:
    start = s.start if s.start is not None else 0
    if s.stop is None:
        stop = length
    elif s.stop < 0:
        stop = (s.stop+length) % length
    else:
        stop = s.stop
    step = s.step or 1
    return range(start, stop, step)


def _make_parents(self):
    if len(self._path) == 1:
        if self._cache is not None:
            pass
        elif isinstance(self._path[0], str):
            self._cache = {}
        else:
            self._cache = []
        return self

    obj = self._cache
    for idx, key in enumerate(self._path[:-1]):
        if not isinstance(obj, collections.abc.Mapping) or self._path[idx+1] in obj:
            try:
                obj = self.normal_get(obj, key)
            except (IndexError, KeyError):
                raise KeyError(self._path[:idx+1])
        elif isinstance(self._path[idx+1], str):
            obj = obj.setdefault(key, {})
        else:
            obj = obj.setdefault(key, [])
    self._cache = obj
    self._path = Path(self._path[-1])
    return self


def normal_check(obj, query, expect=None) -> bool:
    if isinstance(query, Query):
        query = query._query

    if query in [_undefined_, None, _not_found_]:
        return obj
    elif isinstance(query, str):
        if query[0] == '$':
            raise NotImplementedError(query)
            # return _op_tag(query, obj, expect)
        elif isinstance(obj, collections.abc.Mapping):
            return normal_get(obj, query) == expect
        elif hasattr(obj, "_entry"):
            return normal_get(obj._entry, query, _not_found_) == expect
        else:
            raise TypeError(query)
    elif isinstance(query, collections.abc.Mapping):
        return all([normal_check(obj, k, v) for k, v in query.items()])
    elif isinstance(query, collections.abc.Sequence):
        return all([normal_check(obj, k) for k in query])
    else:
        raise NotImplementedError(query)


def normal_erase(obj: typing.Any,  path):

    # for key in path._items[:-1]:
    #     try:

    #     except (IndexError, KeyError):
    #         return False

    obj = normal_get(obj, path._items[:-1])

    if isinstance(obj, (collections.abc.Mapping, collections.abc.Sequence)) and not isinstance(obj, str):
        del obj[path._items[-1]]
        return True
    else:
        return False


def normal_put(obj: typing.Any, path: typing.Any, value, op):
    error_message = None

    if isinstance(obj, Entry):
        obj.child(path).push(value, op)
    elif hasattr(obj.__class__, '__entry__'):
        obj.__entry__().child(path).push(value, op)
    elif not isinstance(obj, (collections.abc.MutableMapping, collections.abc.MutableSequence)):
        error_message = f"Can not put value to {type(obj)}!"
    elif path != 0 and not path:  # is None , empty, [],{},""
        if isinstance(obj, collections.abc.Sequence):
            if op in (Entry.op_tags.extend, Entry.op_tags.update, Entry.op_tags.assign):
                if iterable(value) and not isinstance(value, str):
                    obj.extend(value)
                else:
                    error_message = f"{type(value)} is not iterable!"
            elif op in (Entry.op_tags.append):
                obj.append(value)
            else:
                error_message = False
        elif isinstance(obj, collections.abc.Mapping):
            if op in (Entry.op_tags.update, Entry.op_tags.extend, Entry.op_tags.append):
                if isinstance(value, collections.abc.Mapping):
                    for k, v in value.items():
                        normal_put(obj, k, v, op)
                else:
                    error_message = False
            else:
                error_message = False
        else:
            error_message = f"Can not assign value without key!"
    elif isinstance(path, Path) and len(path) == 1:
        normal_put(obj, path[0], value, op)
    elif isinstance(path, Path) and len(path) > 1:
        for idx, key in enumerate(path._items[:-1]):
            if obj in (None, _not_found_, _undefined_):
                error_message = f"Can not put value to {path[:idx]}"
                break
            elif isinstance(obj, Entry) or hasattr(obj, "__entry__"):
                normal_put(obj, path[idx+1:], value, op)
                break
            else:
                next_obj = normal_get(obj, key)
                if next_obj is not _not_found_:
                    obj = next_obj
                else:
                    if isinstance(path._items[idx+1], str):
                        normal_put(obj, key, {}, Entry.op_tags.assign)
                    else:
                        normal_put(obj, key, [], Entry.op_tags.assign)

                    obj = normal_get(obj, key)

        if obj is not _not_found_:
            normal_put(obj, path[-1], value, op)
    elif path in (None, _undefined_):
        if isinstance(obj, collections.abc.MutableMapping) and isinstance(obj, collections.abc.Mapping) \
                and op in (Entry.op_tags.update, Entry.op_tags.extend, Entry.op_tags.append):
            for k, v in value.items():
                normal_put(obj, k, v, op)
        elif op in (Entry.op_tags.update, Entry.op_tags.extend):
            obj.extend(value)
        elif op in (Entry.op_tags.append):
            obj.append(value)
        else:
            error_message = False
    elif isinstance(path, str) and isinstance(obj, collections.abc.Mapping):
        if op is Entry.op_tags.assign:
            obj[path] = value

        else:
            n_obj = obj.get(path, _not_found_)
            if n_obj is _not_found_ or not isinstance(n_obj, (collections.abc.Mapping, collections.abc.MutableSequence, Entry)):
                if op is Entry.op_tags.append:
                    obj[path] = [value]
                else:
                    obj[path] = value
            else:
                normal_put(n_obj, _undefined_, value, op)
    elif isinstance(path, str) and isinstance(obj, collections.abc.MutableSequence):
        for v in obj:
            normal_put(v, path, value, op)
    elif isinstance(path, int) and isinstance(obj, collections.abc.MutableSequence):
        if path < 0 or path >= len(obj):
            error_message = False
        elif op is Entry.op_tags.assign:
            obj[path] = value
        else:
            normal_put(obj[path], _undefined_, value, op)
    elif isinstance(path, slice) and isinstance(obj, collections.abc.MutableSequence):
        if op is Entry.op_tags.assign:
            obj[path] = value
        else:
            for v in obj[path]:
                normal_put(v, _undefined_, value, op)
    elif isinstance(path, collections.abc.Sequence):
        for i in path:
            normal_put(obj, i, value, op)
    elif isinstance(path, collections.abc.Mapping):
        for i, v in path.items():
            normal_put(obj, i, normal_get(value, v), op)
    elif isinstance(path, Query):
        if isinstance(obj, collections.abc.Sequence):
            for idx, v in enumerate(obj):
                if normal_check(v, path):
                    normal_put(obj, idx, value, op)
        else:
            error_message = f"Only list can accept Query as path!"
    else:
        error_message = False

    if error_message is False:
        error_message = f"Illegal operation!"

    if error_message:
        raise RuntimeError(
            f"Operate Error [{op._name_}]:{error_message} [object={type(obj)} key={path} value={type(value)}]")


def normal_get(obj, path):
    if path is None or (isinstance(path, collections.abc.Sequence) and len(path) == 0) or obj in (None, _not_found_, _undefined_):
        res = obj
    elif isinstance(obj, Entry):
        res = obj.get(path, _not_found_)
    elif hasattr(obj.__class__, "__entry__"):
        res = obj.__entry__().get(path, _not_found_)
    elif isinstance(path, (Path, tuple)):
        res = obj
        for idx, key in enumerate(path):
            if isinstance(res, Entry):
                res = res.get(path[idx:], _not_found_)
                break
            else:
                res = normal_get(res, key)
            if res is _not_found_:
                break
    elif isinstance(path, (Query, collections.abc.Mapping)):  # Mapping => Query
        res = [v for idx, v in normal_filter(obj, path)]
        if len(res) == 0:
            res = _not_found_
    elif isinstance(path, (int, slice)) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        res = obj[path]
    elif isinstance(path, str) and isinstance(obj, collections.abc.Mapping):
        res = obj.get(path, _not_found_)
    elif isinstance(path, set):  # set => Mapping
        res = {k: normal_get(obj, k) for k in path}
    elif isinstance(path, collections.abc.Sequence):  # list => list
        res = [normal_get(obj, k) for k in path]
    elif hasattr(obj, "get") and isinstance(path, str):
        res = obj.get(path, _not_found_)
    # elif isinstance(path, (int, slice)) and isinstance(obj, collections.abc.Mapping):
    #     # res = {k: normal_get(v, path) for k, v in obj.items()}
    #     raise IndexError(f"Can not index Mapping by int or slice! object type={type(obj)} path type={type(path)}")
    # elif isinstance(path, str) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
    #     # res = [normal_get(v, path) for v in obj]
    #     raise IndexError(f"Can not index Sequence by str! object type={type(obj)} path type={type(path)}")
    else:
        raise IndexError(f"Can not index {type(obj)} by {type(path)}")

    return res


class QueryStruct:
    uri: list
    predicate: dict
    sort: list
    limit: int
    offset: int


class Query(object):
    class tags(Flag):
        null = auto()
        read = auto()
        write = auto()
        extend = auto()
        append = auto()
        update = auto()
        erase = auto()
        count = auto()
        equal = auto()

        first_child = auto()
        next = auto()
        parent = auto()

    def __init__(self, path=None, **kwargs) -> None:
        super().__init__()
        self._path: Path = Path(path) if not isinstance(path, Path) else path
        self._query = dict(kwargs)

    def __repr__(self) -> str:
        return f"{self._path}?{'&'.join([f'{k}={v}' for k,v in self._query.items()])}"

    @property
    def path(self) -> Path:
        return self._path

    @property
    def only_first(self) -> bool:
        return self._query.get("onlyu_first", False)

    def apply(self, target, *args, **kargs) -> typing.Any:
        if hasattr(target, "query"):
            return target.query(self, *args, **kargs)
        elif self._path is None:
            return self._apply(target, [])
        else:
            target = normal_get(target, self._path.parent)
            return self._apply(target, self._path[-1])

    def _apply(self, target, path: _TIndex) -> typing.Any:
        if self._path is None:
            return target
        elif isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
            return self.filter(target, self._query)
        elif isinstance(target, collections.abc.Mapping):
            return self.filter(target, self._query)
        else:
            return self.filter(target, self._query)

    def filter(self, obj: collections.abc.Sequence, query: Query) -> typing.Iterator[typing.Tuple[int, typing.Any]]:
        for idx, val in enumerate(obj):
            if normal_check(val, query):
                yield idx, val
                if self.only_first:
                    break

    def update(self, **kwargs):
        self._query.update(kwargs)

    def pull(self, target) -> typing.Any:
        obj = normal_get(self._cache, req)

        if obj is _not_found_:
            if default_value is _undefined_:
                raise IndexError(f"Can not find {req}!")
            elif setdefault:
                self.push(default_value)
            res = default_value
        else:
            self._cache = obj
            self._path.reset()
            res = obj

    def push(self, target) -> bool:

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

    @staticmethod
    def _match(val, predication: collections.abc.Mapping):
        if not isinstance(predication, collections.abc.Mapping):
            predication = {predication: None}

        def do_match(op, value, expected):
            res = False
            if isinstance(op, Entry.op_tag):
                res = Entry._ops[op](value, expected)
            else:
                try:
                    actual, p = Entry._eval_path(
                        value, Entry.normalize_path(op)+[None])
                    res = p is None and (actual == expected)
                except (IndexError, KeyError):
                    res = False

            return res

        success = all([do_match(op, val, args)
                      for op, args in predication.items()])
        return success

    @staticmethod
    def _update(self, key, value):
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
            Return: 返回path中最后一个key，这个key所属于的Tree node

            if force is Ture then:
                当中间node不存在时，根据key的类型创建之，
                    key is str => node is dict
                    key is int => node is list
            else:
                当中间node不存在时，返回最后一个有效node和后续path

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

    def pull(self, path=None, query=_undefined_,  lazy=False, predication=_undefined_, only_first=False, type_hint=_undefined_) -> Any:
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

        obj = self.pull(path, *args, lazy=lazy, **kwargs)
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

    def write(self, /, **kwargs):
        """
            save data to self     
        """
        if isinstance(self, Entry):
            return self.load(self.dump())
        else:
            raise NotImplementedError()

    def read(self, source, /, **kwargs):
        """
            read data from source and merge to cache
        """
        if isinstance(source, collections.abc.Mapping):
            deep_merge_dict(self._cache, source, in_place=True)
        elif isinstance(source, Entry):
            deep_merge_dict(self._cache, source.dump(), in_place=True)
        else:
            raise NotImplementedError()

# def ht_update(self,  query: Optional[_TPath], value, /,  **kwargs) -> Any:
#     if query is not None and len(query) > 0:
#         val = _ht_put(self, query, _not_found_,   **kwargs)
#     else:
#         val = self

#     if isinstance(val, Entry):
#         val.update(None, value,   **kwargs)
#     elif isinstance(val, dict):
#         for k, v in value.items():
#             u = val.setdefault(k, v)
#             if u is not v:
#                 ht_update(u, None, v,  **kwargs)

#     elif hasattr(val, '_entry'):
#         logger.debug(val.__class__)
#         val.update(value, **kwargs)
#     else:
#         _ht_put(self, query, value, assign_if_exists=True, **kwargs)

# def ht_check(self, condition: Mapping) -> bool:
#     def _check_eq(l, r) -> bool:
#         if l is r:
#             return True
#         elif type(l) is not type(r):
#             return False
#         elif isinstance(l, np.ndarray):
#             return np.allclose(l, r)
#         else:
#             return l == r
#     d = [_check_eq(_ht_get(self, k, default_value=_not_found_), v)
#          for k, v in condition.items() if not isinstance(k, str) or k[0] != '_']
#     return all(d)


# def ht_remove(self, query: Optional[_TPath] = None, *args,  **kwargs):

#     if isinstance(self, Entry):
#         return self.remove(query, *args, **kwargs)

#     if len(query) == 0:
#         return False

#     self = _ht_get(self, query[:-1], _not_found_)

#     if self is _not_found_:
#         return
#     elif isinstance(query[-1], str):
#         try:
#             delattr(self, query[-1])
#         except Exception:
#             try:
#                 del self[query[-1]]
#             except Exception:
#                 raise KeyError(f"Can not delete '{query}'")


# def ht_count(self,    *args, default_value=_not_found_, **kwargs) -> int:
#     if isinstance(self, Entry):
#         return self.count(*args, **kwargs)
#     else:
#         self = _ht_get(self, *args, default_value=default_value, **kwargs)
#         if self is None or self is _not_found_:
#             return 0
#         elif isinstance(self, (str, int, float, np.ndarray)):
#             return 1
#         elif isinstance(self, (collections.abc.Sequence, collections.abc.Mapping)):
#             return len(self)
#         else:
#             raise TypeError(f"Not countable! {type(self)}")


# def ht_contains(self, v,  *args,  **kwargs) -> bool:
#     return v in _ht_get(self,  *args,  **kwargs)


# def ht_iter(self, query=None, /,  **kwargs):
#     self = _ht_get(self, query, default_value=_not_found_)
#     if self is _not_found_:
#         yield from []
#     elif isinstance(self, (int, float, np.ndarray)):
#         yield self
#     elif isinstance(self, (collections.abc.Mapping, collections.abc.Sequence)):
#         yield from self
#     elif isinstance(self, Entry):
#         yield from self.iter()
#     else:
#         yield self


# def ht_items(self, query: Optional[_TPath], *args, **kwargs):
#     obj = _ht_get(self, query, *args, **kwargs)
#     if ht_count(obj) == 0:
#         yield from {}
#     elif isinstance(obj, Entry):
#         yield from obj.items()
#     elif isinstance(obj, collections.abc.Mapping):
#         yield from obj.items()
#     elif isinstance(obj, collections.abc.MutableSequence):
#         yield from enumerate(obj)
#     elif isinstance(obj, Entry):
#         yield from obj.items()
#     else:
#         raise TypeError(type(obj))


# def ht_values(self, query: _TPath = None, /, **kwargs):
#     self = _ht_get(self, query, **kwargs)
#     if isinstance(self, collections.abc.Sequence) and not isinstance(self, str):
#         yield from self
#     elif isinstance(self, collections.abc.Mapping):
#         yield from self.values()
#     elif isinstance(self, Entry):
#         yield from self.iter()
#     else:
#         yield self


# def ht_keys(self, query: _TPath = None, /, **kwargs):
#     self = _ht_get(self, query, **kwargs)
#     if isinstance(self, collections.abc.Mapping):
#         yield from self.keys()
#     elif isinstance(self, collections.abc.MutableSequence):
#         yield from range(len(self))
#     else:
#         raise NotImplementedError()


# def ht_compare(first, second) -> bool:
#     if isinstance(first, Entry):
#         first = first.find()
#     if isinstance(second, Entry):
#         second = second.find()
#     return first == second
