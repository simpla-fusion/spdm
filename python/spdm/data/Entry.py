import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import operator
from copy import deepcopy
from enum import Enum, Flag, auto
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from numpy.core.defchararray import count
from numpy.lib.arraysetops import isin

from ..numlib import np
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, serialize

_next_ = object()
_last_ = object()

_T = TypeVar("_T")
_TObject = TypeVar("_TObject")
_TPath = TypeVar("_TPath", int, slice, str,  Sequence)
_TQuery = TypeVar("_TQuery", int,  slice, str, Sequence, Mapping)

_TKey = TypeVar('_TKey', int, str)
_TIndex = TypeVar('_TIndex', int, slice)
_DICT_TYPE_ = dict
_LIST_TYPE_ = list

_TContainer = TypeVar("_TContainer", bound="EntryContainer")


class EntryContainer(Generic[_TObject]):
    __slots__ = "_entry"

    def __init__(self, entry, *args, **kwargs) -> None:
        super().__init__()
        if isinstance(entry, EntryContainer):
            self._entry = entry._entry
        elif not isinstance(entry, Entry):
            self._entry = Entry(entry)
        else:
            self._entry = entry

    def __duplicate__(self) -> _TContainer:
        return self.__class__(self._entry)

    @property
    def empty(self) -> bool:
        return not self._entry.pull(Entry.op_tag.exists)

    def clear(self):
        self._entry.clear()

    def get(self, path: _TQuery = None,  default_value: _T = _undefined_, **kwargs) -> _T:
        return self._entry.get(path, default_value, **kwargs)

    def put(self, path: _TQuery, value: _T, /, **kwargs) -> Tuple[_T, bool]:
        return self._entry.put(path, value,  **kwargs)

    def reset(self, value: _T = None, **kwargs) -> None:
        self._entry.push({Entry.op_tag.assign: value},  **kwargs)

    def update(self,  value: _T,    ** kwargs) -> _T:
        return self._entry.push({Entry.op_tag.update: value},  **kwargs)


PRIMARY_TYPE = (int, float, str, np.ndarray)


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


_TEntry = TypeVar('_TEntry', bound='Entry')


class Entry(object):
    __slots__ = "_cache", "_path"

    class op_tag(Flag):
        # write
        insert = auto()  #
        assign = auto()
        update = auto()
        append = auto()
        erase = auto()
        reset = auto()
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
        # def _op_convert(d):
        #     if isinstance(d, str) and d[0] == '@':
        #         d = Entry.op_tag.__members__.get(d[1:], d)
        #     elif isinstance(d, collections.abc.Mapping):
        #         d = {_op_convert(k): (_op_convert(v) if isinstance(v, collections.abc.Mapping) else v)
        #              for k, v in d.items()}
        #     return d
        # return [_op_convert(d) for d in query]

    def __init__(self, cache=None,   path=None,      **kwargs):
        super().__init__()
        self._cache = None
        self._path = self.normalize_query(path)
        self.enable_cache(cache)

    def duplicate(self) -> _TEntry:
        return self.__class__(self._cache, path=self._path)

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

    def extend(self, rquery: _TQuery) -> _TEntry:
        node = self.duplicate()
        if rquery is not None:
            node._path = node._path + self.normalize_query(rquery)
        return node

    @property
    def parent(self) -> _TEntry:
        if not self._path:
            return self.pull(Entry.op_tag.parent)
        else:
            node = self.duplicate()
            node._path = node._path[:-1]
            return node

    @property
    def first_child(self) -> _TEntry:
        """
            return next brother neighbour
        """
        return self.pull(Entry.op_tag.first_child)

    def __next__(self) -> _TEntry:
        return self.pull(Entry.op_tag.next)

    def reset(self, value=None) -> _TEntry:
        self._cache = value
        self._path = []
        return self

    @property
    def exists(self) -> bool:
        return self.pull(Entry.op_tag.exists)

    @property
    def is_list(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Sequence) and not isinstance(self._cache, str)

    @property
    def is_dict(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Mapping)

    def __len__(self) -> int:
        return self.pull(Entry.op_tag.count)

    @property
    def count(self) -> int:
        return self.pull(Entry.op_tag.count)

    def __contains__(self, k) -> bool:
        return self.pull({Entry.op_tag.exists: k})

    def equal(self, other) -> bool:
        return self.pull({Entry.op_tag.equal: other})

    def get(self, query: _TQuery, default_value: _T = _undefined_, **kwargs) -> _T:
        return self.extend(query).pull(default_value,   **kwargs)

    def put(self, query: _TQuery, value: _T, **kwargs) -> Tuple[_T, bool]:
        return self.extend(query).push(value, **kwargs)

    def append(self, value: _T) -> _T:
        return self.push({Entry.op_tag.append: value})

    def update(self, value: _T) -> _T:
        return self.push({Entry.op_tag.update: value})

    def update_many(self, value=None):
        raise NotImplementedError()

    def erase(self):
        self.push(Entry.op_tag.erase)

    def iter(self):
        obj = self.get(None)

        if isinstance(obj, Entry):
            yield from obj.iter()
        elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
            yield from obj
        else:
            raise NotImplementedError(obj)

    def _op_assign(target, k, v):
        if k is _next_:
            target.append(v)
        else:
            target[k] = v
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

    def _op_append(target, k, v):
        if isinstance(target, collections.abc.Mapping):
            val = target.get(k, _not_found_)
        else:
            val = target[k]

        if val is _not_found_:
            target[k] = _LIST_TYPE_()
            val = target[k]

        val.append(v)
        return v

    def _op_erase(target, k, *args):
        try:
            del target[k]
        except Exception as error:
            success = False
        else:
            success = True
        return success

    def _op_update(target, key, v):
        if isinstance(v, collections.abc.Mapping):
            if key is None:
                val = target
            elif isinstance(target, collections.abc.Mapping):
                val = target.setdefault(key, _DICT_TYPE_())
            else:
                val = target[key]

            for k, v in v.items():
                Entry._ht_put(val, [k], {Entry.op_tag.update: v})
        else:
            if isinstance(target, collections.abc.Mapping):
                val = target.get(key, _not_found_)
            else:
                val = target[key]

            if val is _not_found_:
                target[key] = v
                val = v
        return val

    def _op_check(target, pred=None, *args) -> bool:
        if pred is None:
            return target is not None
        elif isinstance(pred, str) and isinstance(target, collections.abc.Mapping):
            return Entry._op_check(target.get(pred, _not_found_), *args)
        elif isinstance(target, Entry.op_tag):
            return Entry._ops[pred](target, *args)
        elif isinstance(target, collections.abc.Mapping):
            return all([Entry._op_check(target, k, v) for k, v in pred.items()])
        else:
            return target == pred
    _ops = {
        op_tag.insert: _op_insert,
        op_tag.assign: _op_assign,
        op_tag.update: _op_update,
        op_tag.append: _op_append,
        op_tag.erase: _op_erase,  # lambda target, key, v: (del target[key]),

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
    def _ht_get(target, query, default_value: Union[_T,  op_tag] = _undefined_) -> _T:
        """
            Finds an element with key equivalent to key.
            return if key exists return element else return default_value
        """
        if target is _not_found_ or target is None:
            return default_value

        if not query:
            query = [None]
        root = target
        last_idx = len(query)-1
        val = target
        for idx, key in enumerate(query):
            val = _not_found_
            if key is None:
                val = target
            elif isinstance(target, (Entry, )):
                val = target.extend(query[idx:]).pull(default_value)
                break
            elif isinstance(target, EntryContainer):
                val = target._entry.extend(query[idx:]).pull(default_value)
                break
            elif isinstance(target, np.ndarray) and isinstance(key, (int, slice)):
                val = target[key]
            elif isinstance(target, (dict, collections.ChainMap)) and isinstance(key, str):
                val = target.get(key, _not_found_)
            elif isinstance(target, list) and isinstance(key, (int, slice)):
                if isinstance(key, int):
                    val = target[key] if key < len(target) else _not_found_
                elif idx == last_idx and isinstance(key, slice):
                    val = target[key]
                elif isinstance(key, slice):
                    val = [Entry._ht_get(target, [s]+query[idx+1:], default_value=default_value)
                           for s in _slice_to_range(key, len(target))]
                    break
            elif isinstance(target, collections.abc.Sequence) and isinstance(key, (collections.abc.Mapping)):
                val = [v for v in target if Entry._op_check(v, key)]
                if len(val) == 1:
                    val = val[0]
            else:
                raise TypeError(f"{type(target)} {type(key)} {query[:idx+1]}")

            if val is _not_found_:
                break
            target = val

        if default_value is _undefined_ and val is _not_found_:
            val = Entry(target, query[idx:])
        elif isinstance(default_value, Entry.op_tag):
            val = Entry._ops[default_value](val)
        elif isinstance(default_value, collections.abc.Mapping) and any(map(lambda k: isinstance(k, Entry.op_tag), default_value.keys())):
            res = [Entry._ops[k](val, v) for k, v in default_value.items()]
            val = res[0] if len(res) == 1 else res
        elif val is _not_found_:
            val = default_value

        return val

        # elif isinstance(obj, collections.abc.Mapping):
        #     if not isinstance(key, str):
        #         raise TypeError(f"mapping indices must be str, not {type(key).__name__}! \"{query}\"")
        #     tmp = obj.find(key, _not_found_)
        #     obj = tmp
        # elif isinstance(obj, collections.abc.MutableSequence):
        #     if not isinstance(key, (int, slice)):
        #         raise TypeError(
        #             f"list indices must be integers or slices, not {type(key).__name__}! \"{query[:idx+1]}\" {type(obj)}")
        #     elif isinstance(key, int) and isinstance(target._cache, collections.abc.MutableSequence) and key > len(target._cache):
        #         raise IndexError(f"Out of range! {key} > {len(target._cache)}")
        #     obj = obj[key]
        # elif hasattr(obj, "_cache"):
        #     if obj._cache._cache == target._cache and obj._cache._path == query[:idx]:
        #         suffix = query
        #         obj = obj._cache._cache
        #         break
        #     else:
        #         obj = obj._cache.find(query[idx:])
        #     break

        # if rquery is None:
        #     target._cache = obj
        #     target._path = []

    @staticmethod
    def _ht_put(target: Any, query: _TQuery, value: _T, create_if_not_exists=True) -> Tuple[_T, bool]:
        """
            insert if the key does not exist, does nothing if the key exists.
            return value at key
        """
        if not query:
            query = [None]

        last_idx = len(query)-1

        val = target
        idx = 0
        while idx <= last_idx:
            key = query[idx]
            val = _not_found_
            if target is _not_found_ or target is None:
                raise KeyError(query[:idx+1])
            elif isinstance(target, (Entry,   EntryContainer)):
                val = target.put(query[idx:], value)
                break
            # elif isinstance(target, collections.abc.Sequence) and key is _next_:
            #     target.append(None)
            #     query[idx] = len(target)-1
            #     continue
            elif idx == last_idx:
                if isinstance(value, (dict, collections.ChainMap)) and any(map(lambda k: isinstance(k, Entry.op_tag), value.keys())):
                    val = [Entry._ops[k](target, key, v) for k, v in value.items()]
                    if len(val) == 0:
                        val = val[0]
                else:
                    val = Entry._op_assign(target, key, value)
            elif key is None:
                val = target
            elif isinstance(target, (collections.abc.Mapping)) and isinstance(key, str):
                val = target.get(key, _not_found_)
                if val is not _not_found_:
                    pass
                elif create_if_not_exists:
                    if isinstance(query[idx+1], str):
                        val = _DICT_TYPE_()
                        target[key] = val
                    else:
                        val = _LIST_TYPE_()
                        target[key] = val
                else:
                    raise KeyError(query[:idx+1])
            elif isinstance(target, (np.ndarray)) and isinstance(key, (int, slice)):
                val = target[key]
            elif isinstance(target, collections.abc.Sequence) and isinstance(key, (int, slice)):
                val = target[key]
            elif isinstance(target, collections.abc.Sequence) and isinstance(key, (slice, collections.abc.Sequence)):
                if isinstance(key, slice):
                    key = _slice_to_range(slice)

                if isinstance(value, collections.abc.Sequence):
                    val = [Entry._ht_put(target, [j]+query[idx+1:], value[i]) for i, j in enumerate(key)]
                else:
                    val = [Entry._ht_put(target, [j]+query[idx+1:], value) for i, j in enumerate(key)]
                break
            else:
                raise TypeError(f"{type(target)} {type(key)}")

            target = val
            idx = idx+1

        return val

    def pull(self, default_value: Union[op_tag, _T] = _not_found_) -> _T:
        # if isinstance(op, (Entry.op_tag)):
        #     op = {op: None}
        # elif isinstance(op, collections.abc.Mapping) and any(map(lambda k: isinstance(k, Entry.op_tag), op.keys())):
        #     pass
        # else:
        #     op = {Entry.op_tag.fetch: op}

        return self._ht_get(self._cache, self._path,  default_value)

    def push(self,  value: _T) -> Tuple[_T, bool]:
        if value is _undefined_:
            raise ValueError(f"{self._path}")

        if self._cache is None:
            if isinstance(self._path[0], str):
                self._cache = _DICT_TYPE_()
            else:
                self._cache = _LIST_TYPE_()

        if isinstance(value, Entry.op_tag):
            value = {value: None}

        return self._ht_put(self._cache,  self._path, value)


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


class EntryCombiner(Entry):
    def __init__(self, cache, d_list: Sequence = [], /, reducer=None, **kwargs):
        super().__init__(cache,   **kwargs)
        self._reducer = reducer if reducer is not None else operator.__add__
        self._d_list = d_list

    def duplicate(self):
        res = super().duplicate()
        res._reducer = self._reducer
        res._d_list = self._d_list

        return res

    def push(self,  value: _T = _not_found_) -> _T:
        return super().push(value)

    def pull(self,  default_value: _T = _not_found_) -> _T:
        res = default_value

        val = super().pull(_not_found_)

        if val is _not_found_:
            val = Entry._ht_get(self._d_list, [slice(None, None, None)] + self._path, _not_found_)

        if isinstance(val, collections.abc.Sequence):
            val = [d for d in val if (d is not None and d is not _not_found_)]

            if any(map(lambda v: isinstance(v, (Entry, EntryContainer)), val)):
                val = EntryCombiner(val)
            elif len(val) > 0:
                val = functools.reduce(self._reducer, val[1:], val[0])

        return res


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
