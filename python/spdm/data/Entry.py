import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import operator
from ast import copy_location
from copy import deepcopy
from enum import Enum, Flag, auto
from logging import log
from os import initgroups, read
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from numpy.core.defchararray import count
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import insert
from spdm.util.LazyProxy import LazyProxy

from ..numlib import np
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, normalize_path, serialize


class EntryTags(Flag):
    append = auto()
    parent = auto()
    next = auto()
    last = auto()


_parent_ = EntryTags.parent
_next_ = EntryTags.next
_last_ = EntryTags.last
_append_ = EntryTags.append

_T = TypeVar("_T")
_TObject = TypeVar("_TObject")
_TPath = TypeVar("_TPath", int,  slice, str, Sequence, Mapping)

_TKey = TypeVar('_TKey', int, str)
_TIndex = TypeVar('_TIndex', int, slice)
_DICT_TYPE_ = dict
_LIST_TYPE_ = list

_TEntry = TypeVar('_TEntry', bound='Entry')

_TContainer = TypeVar("_TContainer", bound="EntryContainer")


class Entry(object):
    __slots__ = "_cache", "_path"

    PRIMARY_TYPE = (int, float, str, np.ndarray)

    class op_tag(Flag):
        # write
        try_insert = auto()
        assign = auto()
        update = auto()
        append = auto()
        remove = auto()
        reset = auto()
        # read
        find = auto()
        equal = auto()
        count = auto()
        exists = auto()
        contains = auto()
        dump = auto()
        # iter
        next = auto()
        parent = auto()
        first_child = auto()

    def __init__(self, cache=None,   path=None,      **kwargs):
        super().__init__()
        self._cache = cache
        self._path = self.normalize_path(path)
        # self.enable_cache(cache)

    def duplicate(self) -> _TEntry:
        obj = object.__new__(self.__class__)
        obj._cache = self._cache
        obj._path = self._path
        return obj

    def reset(self, value=None) -> _TEntry:
        self._cache = value
        self._path = []
        return self

    def __serialize__(self, *args, **kwargs):
        return self.pull(Entry.op_tag.dump)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cache={type(self._cache)} path={self._path} />"

    @property
    def cache(self) -> Any:
        return self._cache

    @property
    def path(self) -> Sequence:
        return self._path

    @property
    def level(self):
        return len(self._path)

    @property
    def empty(self) -> bool:
        return (self._cache is None and not self._path) or not self.exists

    def predecessor(self) -> _TEntry:
        return NotImplemented

    def successor(self) -> _TEntry:
        return NotImplemented

    @property
    def parent(self) -> _TEntry:
        if not self._path:
            return self.pull(Entry.op_tag.parent)
        else:
            node = self.duplicate()
            node._path = node._path[:-1]
            return node

    @property
    def first_child(self) -> Iterator[_TEntry]:
        """
            return next brother neighbour
        """
        return self.pull(Entry.op_tag.first_child)

    def __iter__(self) -> _TEntry:
        return self.first_child

    def _op_find(target, k, default_value=_undefined_):
        obj, key = Entry._eval_path(target, k, force=False, lazy=False)
        if obj is _not_found_:
            obj = default_value
        elif isinstance(key, (int, str, slice)):
            obj = obj[key]
        elif isinstance(key, list):
            obj = [obj[idx] for idx in key]
        else:
            raise TypeError(type(key))
        return obj
        # if isinstance(target, collections.abc.Mapping):
        # elif isinstance(target, collections.abc.Sequence):
        # else:
        #     raise NotImplementedError(type(target))

    def _op_by_filter(target, pred, op,  *args, on_fail: Callable = _undefined_):
        if not isinstance(target, collections.abc.Sequence):
            raise TypeError(type(target))

        if isinstance(pred, collections.abc.Mapping):
            def pred(val, _cond=pred):
                if not isinstance(val, collections.abc.Mapping):
                    return False
                else:
                    return all([val.get(k, _not_found_) == v for k, v in _cond.items()])

        res = [op(target, idx, *args) for idx, val in enumerate(target) if pred(val)]

        if len(res) == 1:
            res = res[0]
        elif len(res) == 0 and on_fail is not _undefined_:
            res = on_fail(target)
        return res

    def _op_assign(target, path, v):
        target, key = Entry._eval_path(target,  Entry.normalize_path(path), force=True, lazy=False)
        if not isinstance(key, (int, str, slice)):
            raise KeyError(path)
        elif not isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
            raise TypeError(type(target))
        target[key] = v
        return v

    def _op_insert(target, k, v):
        if isinstance(target, collections.abc.Mapping):
            val = target.get(k, _not_found_)
        else:
            val = target[k]

        if val is _not_found_:
            target[k] = v
            val = v

        return val

    def _op_append(target, k,  v):
        if isinstance(target, Entry):
            target = target.get(k,  _LIST_TYPE_())
        else:
            target = target.setdefault(k, _LIST_TYPE_())

        if not isinstance(target, collections.abc.Sequence):
            raise TypeError(type(target))

        target.append(v)

        return v

    def _op_remove(target, k, *args):
        try:
            del target[k]
        except Exception as error:
            success = False
        else:
            success = True
        return success

    def _op_update(target, value):

        if not isinstance(target, collections.abc.Mapping):
            raise TypeError(type(target))
        elif value is None or value is _undefined_:
            return target

        for k, v in value.items():
            tmp = target.setdefault(k, v)

            if tmp is v or v is _undefined_:
                pass
            elif not isinstance(tmp, collections.abc.Mapping):
                target[k] = v
            elif isinstance(v, collections.abc.Mapping):
                Entry._op_update(tmp,  v)
            else:
                raise TypeError(type(v))

        return target

    def _op_try_insert(target, key, v):
        if isinstance(target, collections.abc.Mapping):
            val = target.setdefault(key, v)
        elif isinstance(target, collections.abc.Sequence):
            val = target[key]
            if val is None or val is _not_found_:
                target[key] = v
                val = v
        else:
            raise RuntimeError(type(target))
        return val

    def _op_check(target, pred=None, *args) -> bool:

        if isinstance(pred, Entry.op_tag):
            return Entry._ops[pred](target, *args)
        elif isinstance(pred, collections.abc.Mapping):
            return all([Entry._op_check(Entry._eval_path(target, Entry.normalize_path(k), _not_found_), v) for k, v in pred.items()])
        else:
            return target == pred

    def _op_exist(target, path, *args):
        if path in (None, _not_found_, _undefined_):
            return target not in (None, _not_found_, _undefined_)
        else:
            target, path = Entry._eval_path(target, Entry.normalize_path(path), force=False)
            if isinstance(path, str):
                return path in target
            elif isinstance(path, int):
                return path < len(target)
            else:
                return False

    def _op_equal(target, value):
        return target == value

    def _op_count(target, path):
        if path not in (None, _not_found_, _undefined_):
            target, path = Entry._eval_path(target, Entry.normalize_path(path), force=False)
            try:
                target = target[path]
            except Exception:
                return 0
        return len(target)

    _ops = {
        op_tag.assign: _op_assign,
        op_tag.update: _op_update,
        op_tag.append: _op_append,
        op_tag.remove: _op_remove,
        op_tag.try_insert: _op_try_insert,

        # read
        op_tag.find: _op_find,
        op_tag.equal: lambda target, other: target == other,
        op_tag.count: lambda target, *args: len(target) if target not in (None, _not_found_, _undefined_) else 0,
        op_tag.exists: lambda target, *args: target not in (None, _not_found_, _undefined_),
        op_tag.dump: NotImplemented,

        op_tag.next: None,
        op_tag.parent: None,
        op_tag.first_child: None,
    }

    @staticmethod
    def normalize_path(path):
        if path is None:
            path = []
        elif isinstance(path, str):
            path = [path]
        elif isinstance(path, tuple):
            path = list(path)
        elif not isinstance(path, collections.abc.MutableSequence):
            path = [path]

        path = sum([d.split('.') if isinstance(d, str) else [d] for d in path], [])
        return path

    @staticmethod
    def _predicate(val, predication: collections.abc.Mapping):
        if not isinstance(predication, collections.abc.Mapping):
            predication = {predication: None}

        def do_check(op, value, args):
            if isinstance(op, Entry.op_tag):
                res = Entry._ops[op](value, args)
            else:
                try:
                    res = Entry._eval_pull(value, Entry.normalize_path(op), _not_found_) == args
                except (IndexError, KeyError):
                    res = False

            return res

        return all([do_check(op, val, args) for op, args in predication.items()])

    @staticmethod
    def _update(target, key, value):
        if not isinstance(value, collections.abc.Mapping) \
                or not any(map(lambda k: isinstance(k, Entry.op_tag), value.keys())):
            try:
                target[key] = value
            except (KeyError, IndexError) as error:
                logger.exception(error)
                raise KeyError(key)
        else:
            for op, v in value.items():
                if not isinstance(op, Entry.op_tag):
                    logger.warning(f"Ignore illegal op {op}!")
                Entry._eval_op(op, target, key, v)

        return target

    @staticmethod
    def _eval_path(target, path: list, force=False, lazy=False) -> Tuple[Any, Any]:
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

        if isinstance(target, Entry):
            raise NotImplementedError()
        elif target is _undefined_:
            return _not_found_, None

        elif path is _undefined_:
            return target, None
        elif not isinstance(path, list):
            path = [path]

        last_index = len(path)-1

        val = target
        key = None
        for idx, key in enumerate(path):
            val = _not_found_
            if key is None:
                val = target
            elif target is _not_found_:
                break
            elif isinstance(target, Entry):
                return target.moveto(path[idx:-1], _not_found_),  path[-1]
            elif isinstance(target, EntryContainer):
                return target.get(path[idx:-1], _not_found_), path[-1]
            elif isinstance(target, np.ndarray) and isinstance(key, (int, slice)):
                try:
                    val = target[key]
                except (IndexError, KeyError, TypeError) as error:
                    logger.exception(error)
                    val = _not_found_
            elif isinstance(target, (collections.abc.Mapping)) and isinstance(key, str):
                val = target.get(key, _not_found_)
            elif isinstance(target, (collections.abc.Sequence)) and not isinstance(key, str):
                if key is _next_:
                    target.append(_not_found_)
                    key = len(target)-1
                    val = _not_found_
                elif not isinstance(key, dict):
                    if key is _last_:
                        key = len(target)-1
                    try:
                        val = target[key]
                    except (IndexError, KeyError, TypeError) as error:
                        logger.exception(error)
                        val = _not_found_

                else:
                    iv_list = [[i, v] for i, v in enumerate(target) if Entry._predicate(v, predication=key)]

                    if len(iv_list) == 0:
                        if force:
                            val = deepcopy(key)
                            target.append(val)
                            key = len(target)-1
                    elif len(iv_list) == 1:
                        key, val = iv_list[0]
                    else:
                        key = [i for i, v in iv_list]
                        val = [v for i, v in iv_list]
                        if any(filter(lambda d:  isinstance(d, Entry), val)):
                            val = EntryCombiner(val, path[idx+1:])
            else:
                raise NotImplementedError(f"{type(target)} {type(key)} {path[:idx+1]}")

            if idx < last_index:
                if val is _not_found_:
                    if force:
                        val = _DICT_TYPE_() if isinstance(path[idx+1], str) else _LIST_TYPE_()
                        target[key] = val
                    else:
                        key = path[idx:]
                        break
                target = val

        # if target is _not_found_:
        #     raise KeyError((path, target))
        return target, key

    def moveto(self, rpath: _TPath = None, force=True, lazy=True) -> _TEntry:
        target, key = Entry._eval_path(self._cache,
                                       self._path + self.normalize_path(rpath), lazy=lazy, force=force)
        self._cache = target
        self._path = [key] if not isinstance(key, list) else key

        if not lazy and len(self._path) > 0:
            if self._cache in (None, _not_found_, _undefined_):
                logger.warning(f"'{self._path}' points to a null node")

        return self

    @staticmethod
    def _eval_pull(target, path: list, query, *args, lazy=False):
        """
            if path is found then
                return value
            else
                if lazy then return Entry(target,path) else return _not_found_
        """

        target, key = Entry._eval_path(target, path+[None], force=False)

        if key is None:
            pass
        elif lazy is True:
            return Entry(target, key[:-1])
        else:
            target = _not_found_

        if query is _undefined_ or query is _not_found_:
            val = target
        elif isinstance(query, str) and query[0] == '@':
            query = Entry.op_tag.__members__[query[1:]]
            val = Entry._ops[query](target, *args)
        elif isinstance(query, Entry.op_tag):
            val = Entry._ops[query](target, *args)
        elif not isinstance(query, dict):
            val, key = Entry._eval_path(target, Entry.normalize_path(query)+[None], force=False)
            if key is not None:
                val = query
        else:
            val = {k: Entry._eval_pull(target, Entry.normalize_path(k), v, *args)
                   for k, v in query.items() if not isinstance(k, Entry.op_tag)}
            if len(val) == 0:
                val = [Entry._ops[op](target, v, *args)
                       for op, v in query.items() if isinstance(op, Entry.op_tag)]

                if len(val) == 1:
                    val = val[0]
                elif len(val) == 0:
                    val = _not_found_

        return val

    def pull(self, path=None, query=_undefined_,  lazy=False, predication=_undefined_, only_first=False) -> _T:
        if isinstance(path, (Entry.op_tag, collections.abc.Mapping)) and query is _undefined_:
            query = path
            path = None

        path = self._path+Entry.normalize_path(path)

        if predication is _undefined_:
            return Entry._eval_pull(self._cache, path, query, lazy=lazy)
        else:
            target, key = Entry._eval_path(self._cache, path+[None], force=False)
            if key is not None:
                val = Entry._eval_pull(_not_found_, [],  query)
            elif not isinstance(target, list):
                raise TypeError(f"If predication is defined, target must be list! {type(target)}")
            elif only_first:
                try:
                    target = next(filter(lambda d: Entry._predicate(d, predication), target))
                except StopIteration:
                    target = _not_found_

                val = Entry._eval_pull(target, [],  query)
            else:
                val = [Entry._eval_pull(d, [],  query) for d in target if Entry._predicate(d, predication)]
                if len(val) == 0:
                    val = Entry._eval_pull(_not_found_, [],  query)
                elif len(val) == 1:
                    val = val[0]

        return val

    @staticmethod
    def _eval_push(target, path: list, query=_undefined_, *args):
        if path is _undefined_:
            path = []
        elif not isinstance(path, list):
            path = [path]
        if not isinstance(query, np.ndarray) and query is _undefined_:
            val = query
        elif isinstance(query, dict):
            target, p = Entry._eval_path(target, path+[""], force=True)
            if p != "":
                raise KeyError(path)
            val_changed = [Entry._eval_push(target, [k], v, *args)
                           for k, v in query.items() if not isinstance(k, Entry.op_tag)]
            val = [Entry._ops[op](target, v, *args)
                   for op, v in query.items() if isinstance(op, Entry.op_tag)]

            if len(val) == 1:
                val = val[0]
        else:
            target, p = Entry._eval_path(target, path, force=True)
            if target is _not_found_:
                raise KeyError(path)
            if isinstance(query, Entry.op_tag):
                val = Entry._ops[query](target, p, *args)
            elif isinstance(query, str) and query[0] == '@':
                query = Entry.op_tag.__members__[query[1:]]
                val = Entry._ops[query](target, p, *args)
            else:
                val = query
                try:
                    target[p] = val
                except (KeyError, IndexError) as error:
                    logger.exception(error)
                    raise KeyError(path)

        return val

    def push(self, path, value=None, predication=_undefined_, only_first=False) -> _T:
        if value is None:
            value = path
            path = None

        path = self._path + Entry.normalize_path(path)
        if self._cache is _not_found_ or self._cache is _undefined_:
            if len(path) > 0 and isinstance(path[0], str):
                self._cache = _DICT_TYPE_()
            else:
                self._cache = _LIST_TYPE_()

        if predication is _undefined_:
            target, key = Entry._eval_path(self._cache, path, force=True)

            if target is _not_found_ or isinstance(key, list):
                raise KeyError(path)
            val = Entry._eval_push(target, [key] if key is not None else [], value)
        else:
            target, key = Entry._eval_path(self._cache, path+[None], force=True)
            if key is not None or target is _not_found_:
                raise KeyError(path)
            elif not isinstance(target, list):
                raise TypeError(f"If predication is defined, target must be list! {type(target)}")
            elif only_first:
                try:
                    target = next(filter(lambda d: Entry._predicate(d, predication), target))
                except StopIteration:
                    val = _not_found_
                else:
                    val = Entry._eval_push(target, [], value)
            else:
                val = [Entry._eval_push(d, [], value) for d in target if Entry._predicate(d, predication)]
                if len(val) == 0:
                    val = _not_found_

        return val

    def remove(self, query):
        return self.push(query, Entry.op_tag.remove)

    def count(self, query: _TPath = None):
        return self.pull(query, Entry.op_tag.count)

    def exist(self, query: _TPath = None):
        return self.pull(query, Entry.op_tag.exists)

    def get(self, path, default_value=_undefined_, *args, **kwargs) -> Any:
        obj = self.pull(path, *args,  **kwargs)
        if obj is not _not_found_:
            return obj
        elif default_value is not _undefined_:
            return default_value
        else:
            raise KeyError((path, default_value))

    def put(self, *args, **kwargs) -> Any:
        return self.push(*args, **kwargs)


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


class EntryCombiner(Entry):
    def __init__(self,  d_list: Sequence = [], /,
                 default_value=_undefined_,
                 reducer=_undefined_,
                 partition=_undefined_, **kwargs):
        super().__init__(default_value, **kwargs)
        self._reducer = reducer if reducer is not _undefined_ else operator.__add__
        self._partition = partition if partition is not _undefined_ else operator.__add__
        self._d_list = d_list

    def duplicate(self):
        res = super().duplicate()
        res._reducer = self._reducer
        res._d_list = self._d_list

        return res

    def __len__(self):
        return len(self._d_list)

    def __iter__(self) -> Iterator[Entry]:
        raise NotImplementedError()

    def push(self,   *args, **kwargs) -> _T:
        return super().push(*args, **kwargs)

    def pull(self, path=None, query=_undefined_, lazy=False, predication=_undefined_, only_first=False) -> _T:

        val = super().pull(path, lazy=lazy, predication=predication, only_first=only_first)

        if predication is not _undefined_:
            logger.warning("NotImplemented")

        if val is not _not_found_:
            return val

        val = [Entry._eval_path(d, self._path+Entry.normalize_path(path)+[None],  force=False)
               for d in self._d_list]
        val = [d for d, p in val if p is None and d is not _not_found_]

        if isinstance(val, collections.abc.Sequence):
            val = [d for d in val if (d is not _not_found_ and d is not None)]

            if any(map(lambda v: isinstance(v, (Entry, EntryContainer, collections.abc.Mapping)), val)):
                val = EntryCombiner(val)
            elif len(val) == 1:
                val = val[0]
            elif len(val) > 1:
                val = functools.reduce(self._reducer, val[1:], val[0])
            else:
                val = _not_found_

        # val = Entry._eval_pull(val, [], query, lazy=False)

        return val


class EntryWrapper(Entry):

    def __init__(self,  *sources, **kwargs):
        super().__init__(sources, **kwargs)

    def pull(self, default_value=_undefined_, **kwargs):
        res = next(filter(lambda d: d is not _not_found_, map(
            lambda d: _ht_get(d, self._path, default_value=_not_found_), self._cache)), default_value)
        if res is _undefined_:
            res = EntryWrapper(self._cache, path=self._path)
        return res

    def push(self, value, *args, **kwargs):
        return _ht_put(self._cache[0], self._path, value, *args, **kwargs)

    def iter(self):
        for d in self._cache:
            yield from d.iter()

    def items(self):
        for k in self.keys():
            yield k, self.find(k)

    def keys(self):
        k = set()
        for d in self._cache:
            k.add(d.keys())
        yield from k

    def values(self):
        for k in self.keys():
            yield self.find(k)


class EntryIterator(Iterator[_TObject]):
    def __init__(self, holder, index: int = 0, predicate: Callable[[_TObject], bool] = None) -> None:
        super().__init__()
        self._pos = index
        self._predicate = predicate

    def __next__(self) -> Iterator[_TObject]:
        return super().__next__()


class EntryContainer:
    __slots__ = "_entry"

    def __init__(self, entry) -> None:
        super().__init__()
        if isinstance(entry, EntryContainer):
            self._entry = entry._entry
        elif not isinstance(entry, Entry):
            self._entry = Entry(entry)
        else:
            self._entry = entry

    def _duplicate(self) -> _TContainer:
        return self.__class__(self._entry)

    def _pre_process(self, value: Any, *args, **kwargs) -> Any:
        return value

    def _post_process(self, value: _T,   *args,  **kwargs) -> _T:
        return value

    def __ior__(self,  value: _T) -> _T:
        return self._entry.push({Entry.op_tag.update: value})

    def clear(self):
        self._entry.push(Entry.op_tag.reset)

    def remove(self, query: _TPath = None) -> bool:
        return self._entry.push({Entry.op_tag.remove: query})

    def update(self, value: _T, **kwargs) -> _T:
        return self._entry.push({Entry.op_tag.update: value}, **kwargs)

    def find(self, query: _TPath, **kwargs) -> _TObject:
        return self._entry.pull({Entry.op_tag.find: query},  **kwargs)

    def try_insert(self, query: _TPath, value: _T, **kwargs) -> _T:
        return self._entry.push({Entry.op_tag.try_insert: {query: value}},  **kwargs)

    def count(self, query: _TPath, **kwargs) -> int:
        return self._entry.pull({Entry.op_tag.count: query}, **kwargs)

    def dump(self) -> Union[Sequence, Mapping]:
        return self._entry.pull(Entry.op_tag.dump)

    def put(self, path: _TPath, value, *args, **kwargs) -> _TObject:
        return self._entry.put(path, value, *args, **kwargs)

    def get(self, path: _TPath, *args, **kwargs) -> _TObject:
        return self._entry.get(path, *args, **kwargs)

    def equal(self, path: _TPath, other) -> bool:
        return self._entry.pull(path, {Entry.op_tag.equal: other})

    def __setitem__(self, path: _TPath, value: _T) -> _T:
        return self._entry.push(path, self._pre_process(value))

    def __getitem__(self, path: _TPath) -> Union[_TEntry, Any]:
        return self._post_process(self._entry.pull(path, query=None, lazy=True), path=path)

    def __delitem__(self, path: _TPath) -> bool:
        return self._entry.push({Entry.op_tag.remove: path})

    def __contains__(self, path: _TPath) -> bool:
        return self._entry.pull({Entry.op_tag.contains: path})

    def __len__(self) -> int:
        return self._entry.pull(Entry.op_tag.count)

    def __iter__(self) -> Iterator[_T]:
        for idx, obj in enumerate(self._entry.first_child()):
            yield self._post_process(obj, idx)


def as_dataclass(dclass, obj, default_value=None):
    if dclass is dataclasses._MISSING_TYPE:
        return obj

    if hasattr(obj, '_entry'):
        obj = obj._entry
    if obj is None:
        obj = default_value

    if obj is None or not dataclasses.is_dataclass(dclass) or isinstance(obj, dclass):
        pass
    # elif getattr(obj, 'empty', False):
    #     obj = None
    elif dclass is np.ndarray:
        obj = np.asarray(obj)
    elif hasattr(obj.__class__, 'get'):
        obj = dclass(**{f.name: as_dataclass(f.type, obj.get(f.name,  f.default if f.default is not dataclasses.MISSING else None))
                        for f in dataclasses.fields(dclass)})
    elif isinstance(obj, collections.abc.Sequence):
        obj = dclass(*obj)
    else:
        try:
            obj = dclass(obj)
        except Exception as error:
            logger.debug((type(obj), dclass))
            raise error
    return obj


def convert_from_entry(cls, obj, *args, **kwargs):
    origin_type = getattr(cls, '__origin__', cls)
    if dataclasses.is_dataclass(origin_type):
        obj = as_dataclass(origin_type, obj)
    elif inspect.isclass(origin_type):
        obj = cls(obj, *args, **kwargs)
    elif callable(cls) is not None:
        obj = cls(obj, *args, **kwargs)

    return obj


# def ht_update(target,  query: Optional[_TPath], value, /,  **kwargs) -> Any:
#     if query is not None and len(query) > 0:
#         val = _ht_put(target, query, _not_found_,   **kwargs)
#     else:
#         val = target

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
#         _ht_put(target, query, value, assign_if_exists=True, **kwargs)

# def ht_check(target, condition: Mapping) -> bool:
#     def _check_eq(l, r) -> bool:
#         if l is r:
#             return True
#         elif type(l) is not type(r):
#             return False
#         elif isinstance(l, np.ndarray):
#             return np.allclose(l, r)
#         else:
#             return l == r
#     d = [_check_eq(_ht_get(target, k, default_value=_not_found_), v)
#          for k, v in condition.items() if not isinstance(k, str) or k[0] != '_']
#     return all(d)


# def ht_remove(target, query: Optional[_TPath] = None, *args,  **kwargs):

#     if isinstance(target, Entry):
#         return target.remove(query, *args, **kwargs)

#     if len(query) == 0:
#         return False

#     target = _ht_get(target, query[:-1], _not_found_)

#     if target is _not_found_:
#         return
#     elif isinstance(query[-1], str):
#         try:
#             delattr(target, query[-1])
#         except Exception:
#             try:
#                 del target[query[-1]]
#             except Exception:
#                 raise KeyError(f"Can not delete '{query}'")


# def ht_count(target,    *args, default_value=_not_found_, **kwargs) -> int:
#     if isinstance(target, Entry):
#         return target.count(*args, **kwargs)
#     else:
#         target = _ht_get(target, *args, default_value=default_value, **kwargs)
#         if target is None or target is _not_found_:
#             return 0
#         elif isinstance(target, (str, int, float, np.ndarray)):
#             return 1
#         elif isinstance(target, (collections.abc.Sequence, collections.abc.Mapping)):
#             return len(target)
#         else:
#             raise TypeError(f"Not countable! {type(target)}")


# def ht_contains(target, v,  *args,  **kwargs) -> bool:
#     return v in _ht_get(target,  *args,  **kwargs)


# def ht_iter(target, query=None, /,  **kwargs):
#     target = _ht_get(target, query, default_value=_not_found_)
#     if target is _not_found_:
#         yield from []
#     elif isinstance(target, (int, float, np.ndarray)):
#         yield target
#     elif isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
#         yield from target
#     elif isinstance(target, Entry):
#         yield from target.iter()
#     else:
#         yield target


# def ht_items(target, query: Optional[_TPath], *args, **kwargs):
#     obj = _ht_get(target, query, *args, **kwargs)
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


# def ht_values(target, query: _TPath = None, /, **kwargs):
#     target = _ht_get(target, query, **kwargs)
#     if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
#         yield from target
#     elif isinstance(target, collections.abc.Mapping):
#         yield from target.values()
#     elif isinstance(target, Entry):
#         yield from target.iter()
#     else:
#         yield target


# def ht_keys(target, query: _TPath = None, /, **kwargs):
#     target = _ht_get(target, query, **kwargs)
#     if isinstance(target, collections.abc.Mapping):
#         yield from target.keys()
#     elif isinstance(target, collections.abc.MutableSequence):
#         yield from range(len(target))
#     else:
#         raise NotImplementedError()


# def ht_compare(first, second) -> bool:
#     if isinstance(first, Entry):
#         first = first.find()
#     if isinstance(second, Entry):
#         second = second.find()
#     return first == second
