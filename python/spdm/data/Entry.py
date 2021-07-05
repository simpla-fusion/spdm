import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
from logging import log
import operator
from copy import deepcopy
from enum import Enum, Flag, auto
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from numpy.core.defchararray import count
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import insert
from spdm.util.LazyProxy import LazyProxy

from ..numlib import np
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, serialize


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
_TPath = TypeVar("_TPath", int, slice, str,  Sequence)
_TQuery = TypeVar("_TQuery", int,  slice, str, Sequence, Mapping)

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
        insert = auto()  #
        assign = auto()
        update = auto()
        append = auto()
        erase = auto()
        reset = auto()
        try_insert = auto()
        # read
        fetch = auto()
        equal = auto()
        count = auto()
        exists = auto()
        dump = auto()
        # iter
        next = auto()
        parent = auto()
        first_child = auto()

    @staticmethod
    def normalize_query(query):
        if query is None:
            query = []
        elif isinstance(query, str):
            query = [query]
        elif isinstance(query, tuple):
            query = list(query)
        elif not isinstance(query, collections.abc.MutableSequence):
            query = [query]

        query = sum([d.split('.') if isinstance(d, str) else [d] for d in query], [])
        return query

    def __init__(self, cache=None,   path=None,      **kwargs):
        super().__init__()
        self._cache = cache
        self._path = self.normalize_query(path)
        # self.enable_cache(cache)

    def duplicate(self) -> _TEntry:
        obj = object.__new__(self.__class__)
        obj._cache = self._cache
        obj._path = self._path
        return obj

    def __serialize__(self, *args, **kwargs):
        return NotImplemented

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cache={type(self._cache)} path={self._path} />"

    def enable_cache(self, cache: Union[bool, Mapping] = None) -> bool:
        if cache is None:
            pass
        elif cache is False:
            self._cache = None
        elif cache is True:
            if self._cache is None:
                self._cache = _DICT_TYPE_()
        else:
            self._cache = cache

        return self._cache is not None

    @property
    def cache(self) -> Any:
        return self._cache

    @property
    def path(self) -> Sequence:
        return self._path

    @property
    def is_relative(self) -> bool:
        return len(self._path) > 0

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
            return self.pull(op=Entry.op_tag.parent)
        else:
            node = self.duplicate()
            node._path = node._path[:-1]
            return node

    @property
    def first_child(self) -> _TEntry:
        """
            return next brother neighbour
        """
        return self.pull(op=Entry.op_tag.first_child)

    def __next__(self) -> _TEntry:
        return self.pull(op=Entry.op_tag.next)

    def reset(self, value=None) -> _TEntry:
        self._cache = value
        self._path = []
        return self

    @property
    def is_list(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Sequence) and not isinstance(self._cache, str)

    @property
    def is_dict(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Mapping)

    @property
    def count(self) -> int:
        return self.pull(Entry.op_tag.count)

    def contains(self, k) -> bool:
        return self.pull({Entry.op_tag.exists: k})

    @property
    def exists(self) -> bool:
        return self.pull(op=Entry.op_tag.exists)

    def equal(self, other) -> bool:
        return self.pull({Entry.op_tag.equal: other})

    def append(self, value: _T) -> _T:
        return self.push({Entry.op_tag.append: value})

    def update(self, value: _T, predication=_undefined_, only_first=False) -> _T:
        return self.push({Entry.op_tag.update: value}, predication=predication, only_first=only_first)

    def find(self, predication, only_first=False) -> _T:
        return self.pull(predication=predication, only_first=only_first)

    def erase(self):
        self.push(Entry.op_tag.erase)

    def iter(self):
        obj = self.pull()

        if isinstance(obj, Entry):
            yield from obj.iter()
        elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
            yield from obj
        else:
            raise NotImplementedError(obj)

    def _op_by_filter(self, pred, op,  *args, on_fail: Callable = _undefined_):
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

    def _op_assign(self, k, v):
        if k is _next_:
            target.append(v)
        else:
            target[k] = v
        return v

    def _op_insert(self, k, v):
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

    def _op_erase(self, k, *args):
        try:
            del target[k]
        except Exception as error:
            success = False
        else:
            success = True
        return success

    def _op_update(target, k, value):
        if k not in (None, _not_found_, _undefined_):
            try:
                tmp = target[k]
            except (IndexError, KeyError):
                tmp = None

            if tmp is None or not isinstance(value, collections.abc.Mapping):
                target[k] = value
                return value
            else:
                target = tmp

        if not isinstance(target, collections.abc.Mapping):
            raise TypeError(type(target))

        for k, v in value.items():
            tmp = target.setdefault(k, v)

            if tmp is v:
                pass
            elif not isinstance(tmp, collections.abc.Mapping):
                target[k] = v
            else:
                Entry._op_update(tmp, None, v)

        return target

    def _op_try_insert(self, key, v):
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

    def _op_check(self, pred=None, *args) -> bool:

        if isinstance(pred, Entry.op_tag):
            return Entry._ops[pred](target, *args)
        elif isinstance(pred, collections.abc.Mapping):
            return all([Entry._op_check(Entry._eval_path(target, Entry.normalize_query(k), _not_found_), v) for k, v in pred.items()])
        else:
            return target == pred

    _ops = {
        op_tag.insert: _op_insert,
        op_tag.assign: _op_assign,
        op_tag.update: _op_update,
        op_tag.append: _op_append,
        op_tag.erase: _op_erase,  # lambda target, key, v: (del target[key]),
        op_tag.try_insert: _op_try_insert,  # lambda target, key, v: (del target[key]),

        # read
        op_tag.fetch: lambda v, default_value: v if v is not _not_found_ else default_value,
        op_tag.equal: lambda v, other: v == other,
        op_tag.count: lambda v, *other: len(v) if isinstance(v, (collections.abc.Sequence, collections.abc.Mapping, np.ndarray)) else 1,
        op_tag.exists: lambda v, *other: v is not _not_found_,
        op_tag.dump: NotImplemented,

        op_tag.next: None,
        op_tag.parent: None,
        op_tag.first_child: None,
    }

    @staticmethod
    def _check(val, predication: collections.abc.Mapping, only_first=True) -> bool:
        return False

    @staticmethod
    def _apply_filter(val, predication: collections.abc.Mapping, only_first=True):
        if not isinstance(val, (list, Entry, EntryContainer)):
            return val
        elif isinstance(predication, (int, slice)):
            return val[predication]
        elif not isinstance(predication, collections.abc.Mapping):
            return val

        def _filter(d):
            return all([Entry._eval_path(d, Entry.normalize_query(k), read_only=True) == v for k, v in predication.items()])

        if only_first:
            try:
                val = next(filter(_filter, val))
            except StopIteration:
                val = _not_found_
        else:
            val = list(filter(_filter, val))
        return val

    @staticmethod
    def _apply_op(op: Union[collections.abc.Mapping, op_tag], target, *args):
        if op in (None, _undefined_, _not_found_):
            return target

        if isinstance(op, Entry.op_tag):
            return Entry._ops[op](target, *args)
        elif isinstance(op, collections.abc.Mapping) and len(op) > 0:
            val = [Entry._ops[k](target, v, *args) for k, v in op.items() if isinstance(k, Entry.op_tag)]
            if len(val) == 1:
                val = val[0]
        elif op is not _undefined_:
            raise NotImplementedError(op)

        return val

    @staticmethod
    def _eval_path(target, path, default_value=_undefined_, read_only=False, force=_undefined_) -> _TEntry:
        """
            Finds an element with key equivalent to key.
            return if key exists return element else return default_value
        """
        if path is None or not path:
            return target
        elif not isinstance(path, list):
            path = [path]

        if force is _undefined_:
            force = not read_only

        n_path = len(path)

        val = target

        for idx in range(n_path):
            key = path[idx]
            val = _not_found_
            if key is None:
                val = target
            elif isinstance(target, Entry):
                return target.moveto(path[idx:], default_value=default_value, force=force, last_idx=last_idx)
            elif isinstance(target, EntryContainer):
                return target.get(path[idx:-1], default_value=default_value, force=force,  last_idx=last_idx)
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
                    val = Entry._apply_filter(target, key)
                    if not isinstance(val, list):
                        pass
                    elif len(val) == 0 or val is _not_found_ or val is None:
                        if not force:
                            val = _not_found_
                        else:
                            val = deepcopy(key)
                            target.append(val)
                    elif any([not isinstance(d, Entry.PRIMARY_TYPE) for d in val]):
                        val = EntryCombiner(val)
            else:
                raise NotImplementedError(f"{type(target)} {type(key)} {path[:idx+1]}")

            if val is _not_found_:
                if not force:
                    if read_only:
                        raise IndexError(path[:idx+1])
                    else:
                        return Entry(target, path[idx:])
                elif idx < n_path-1:
                    val = _DICT_TYPE_() if isinstance(path[idx+1], str) else _LIST_TYPE_()
                    target[key] = val
                elif default_value is not _undefined_:
                    val = default_value
                    target[key] = val

            if idx == n_path-1 and not read_only and isinstance(key, (str, int)) and \
                    (isinstance(val, Entry.PRIMARY_TYPE) or val in (_not_found_, None, _undefined_)):
                target = Entry(target, key)
            else:
                target = val

        return target

    def moveto(self, rpath: _TQuery = None, lazy=True, default_value=_not_found_, force=False) -> _TEntry:
        res = self.duplicate()
        if rpath is not None:
            res._path = res._path + self.normalize_query(rpath)
        if not lazy:
            res._cache = Entry._eval_path(res._cache, res._path, default_value=default_value, force=force)
            res._path = []
        return res

    def pull(self, op=_undefined_, default_value=_undefined_, lazy=False, predication=_undefined_, only_first=False, ** kwargs) -> _T:
        try:
            val = Entry._eval_path(self._cache, self._path, read_only=True)
        except (IndexError, KeyError) as error:
            if lazy:
                val = Entry(self._cache, self._path)
            else:
                val = _not_found_

        if predication is not _undefined_ and isinstance(val, list):
            val = Entry._apply_filter(val, predication, only_first=only_first)

        if op is not _undefined_:
            val = Entry._apply_op(op, val)

        if val is not _not_found_:
            return val
        elif default_value is not _undefined_:
            return default_value
        elif lazy:
            return Entry(self._cache, self._path)
        else:
            raise KeyError(self._path)

    def push(self,  value: _T = None, predication=_undefined_, only_first=False,  **kwargs) -> _T:

        if predication not in (_undefined_, _not_found_, None):
            raise NotImplementedError()

        target = Entry._eval_path(self._cache, self._path,  read_only=False)

        if isinstance(target, Entry):
            key = target._path[0]
            if len(target._path) > 1:
                raise KeyError(target._path)
            target = target._cache
        else:
            key = None

        if isinstance(value, Entry.op_tag):
            value = {value: None}
        val = _not_found_
        if isinstance(value, collections.abc.Mapping):
            val = [Entry._apply_op(op, target, key, v) for op, v in value.items() if isinstance(op, Entry.op_tag)]
            if len(val) == 0:
                val = _not_found_
            elif len(val) == 1:
                val = val[0]

        if val is not _not_found_:
            pass
        elif isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
            target[key] = value
            val = value
        else:
            raise TypeError(type(target))
        return val

    def put(self, query, *args, **kwargs) -> Any:
        return self.moveto(query).push(*args, **kwargs)

    def get(self, query, *args, **kwargs) -> Any:
        return self.moveto(query).pull(*args, **kwargs)

    def has_child(self, member_name) -> bool:
        return self._eval_path(self._cache, self._path+[member_name], op=Entry.op_tag.exists)

    def flush(self, v):
        self._cache = self.push(v, op=Entry.op_tag.try_insert)
        self._path = []


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

    def clear(self):
        self._entry.push(Entry.op_tag.reset)

    def _pre_process(self, value: Any, *args, **kwargs) -> Any:
        return value

    def _post_process(self, value: _T,   *args,  **kwargs) -> _T:
        return value

    def _remove(self, path: _TQuery, /, **kwargs) -> None:
        return self._entry.moveto(path).push(Entry.op_tag.erase, **kwargs)

    def _reset(self, value: _T = None, **kwargs) -> None:
        self._entry.push(value,  **kwargs)

    def __ior__(self,  value: _T) -> _T:
        return self._entry.push({Entry.op_tag.update: value})

    def get(self, query, default_value=_undefined_) -> _TObject:
        return self._entry.moveto(query).pull(default_value)

    def __setitem__(self, query: _TQuery, value: _T) -> _T:
        return self._entry.moveto(query, force=True).push(self._pre_process(value))

    def __getitem__(self, query: _TQuery) -> Union[_TEntry, Any]:
        return self._post_process(self._entry.moveto(query).pull(), query=query)

    def __delitem__(self, query: _TQuery) -> bool:
        return self._entry.moveto(query).erase()

    def __contains__(self, query: _TQuery) -> bool:
        return self._entry.moveto(query).pull(Entry.op_tag.exists)

    def __len__(self) -> int:
        return self._entry.pull(Entry.op_tag.count)

    def __iter__(self) -> Iterator[_T]:
        for idx, obj in enumerate(self._entry.iter()):
            yield self._post_process(obj, idx)

    def __eq__(self, other) -> bool:
        return self._entry.pull({Entry.op_tag.equal: other})

    def __bool__(self) -> bool:
        return self.__len__() > 0

    def _as_dict(self) -> Mapping:
        return {k: self._post_process(v) for k, v in self._entry.items()}

    def _as_list(self) -> Sequence:
        return [self._post_process(v) for v in self._entry.values()]


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
    def __init__(self,  d_list: Sequence = [], /, default_value=None, reducer=_undefined_,  partition=_undefined_, **kwargs):
        super().__init__(default_value,   **kwargs)
        self._reducer = reducer if reducer is not _undefined_ else operator.__add__
        self._partition = partition if partition is not _undefined_ else operator.__add__
        self._d_list = d_list

    def duplicate(self):
        res = super().duplicate()
        res._reducer = self._reducer
        res._d_list = self._d_list

        return res

    def iter(self):
        raise NotImplementedError()

    def push(self,  value: _T = _not_found_, **kwargs) -> _T:
        return super().push(value, **kwargs)

    def pull(self,  default_value: _T = _not_found_, **kwargs) -> _T:

        val = super().pull(default_value=_not_found_)

        if val is _not_found_:
            val = [Entry._eval_path(d, self._path, default_value=_not_found_, **kwargs) for d in self._d_list]

        if isinstance(val, collections.abc.Sequence):
            val = [d for d in val if (d is not None and d is not _not_found_)]

            if any(map(lambda v: isinstance(v, (Entry, EntryContainer, collections.abc.Mapping)), val)):
                val = EntryCombiner(val)
            elif len(val) == 1:
                val = val[0]
            elif len(val) > 1:
                val = functools.reduce(self._reducer, val[1:], val[0])
            else:
                val = _not_found_

        if val is _not_found_:
            val = default_value
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


# def ht_update(target,  query: Optional[_TQuery], value, /,  **kwargs) -> Any:
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


# def ht_erase(target, query: Optional[_TQuery] = None, *args,  **kwargs):

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


# def ht_items(target, query: Optional[_TQuery], *args, **kwargs):
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


# def ht_values(target, query: _TQuery = None, /, **kwargs):
#     target = _ht_get(target, query, **kwargs)
#     if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
#         yield from target
#     elif isinstance(target, collections.abc.Mapping):
#         yield from target.values()
#     elif isinstance(target, Entry):
#         yield from target.iter()
#     else:
#         yield target


# def ht_keys(target, query: _TQuery = None, /, **kwargs):
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
