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
from numpy.lib.function_base import insert


from ..numlib import np
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, serialize


class EntryTags(Flag):
    next = auto()
    last = auto()


_next_ = EntryTags.next
_last_ = EntryTags.last

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

    @property
    def empty(self) -> bool:
        return not self._entry.pull(Entry.op_tag.exists)

    def clear(self):
        self._entry.clear()

    def get(self, path: _TQuery = None,  default_value: _T = _undefined_, **kwargs) -> _T:
        return self._entry.get(path, default_value, **kwargs)

    def put(self, path: _TQuery, value: _T, /, **kwargs) -> Tuple[_T, bool]:
        return self._entry.put(path, value,  **kwargs)

    def remove(self, path: _TQuery, /, **kwargs) -> None:
        return self._entry.put(path, None, op=Entry.op_tag.erase, **kwargs)

    def reset(self, value: _T = None, **kwargs) -> None:
        self._entry.push(value,  **kwargs)

    def update(self,  value: _T,  predication: Mapping = _undefined_, ** kwargs) -> _T:
        return self._entry.update(value,  predication=predication,  **kwargs)

    def find(self,  *args, ** kwargs) -> _T:
        return self._entry.find(*args,   **kwargs)


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

    def extend(self, rquery: _TQuery) -> _TEntry:
        node = self.duplicate()
        if rquery is not None:
            node._path = node._path + self.normalize_query(rquery)
        return node

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
    def exists(self) -> bool:
        return self.pull(op=Entry.op_tag.exists)

    @property
    def is_list(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Sequence) and not isinstance(self._cache, str)

    @property
    def is_dict(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Mapping)

    def __len__(self) -> int:
        return self.pull(op=Entry.op_tag.count)

    @property
    def count(self) -> int:
        return self.pull(op=Entry.op_tag.count)

    def __contains__(self, k) -> bool:
        return self.pull(op={Entry.op_tag.exists: k})

    def equal(self, other) -> bool:
        return self.pull(op={Entry.op_tag.equal: other})

    def get(self, query: _TQuery, default_value: _T = _undefined_, **kwargs) -> _T:
        return self.extend(query).pull(default_value,   **kwargs)

    def put(self, query: _TQuery, value: _T, **kwargs) -> Tuple[_T, bool]:
        return self.extend(query).push(value, **kwargs)

    def append(self, value: _T) -> _T:
        return self.push(value, op=Entry.op_tag.append)

    def update(self, value: _T, predication=None, only_first=False) -> _T:
        return self.push(value, op=Entry.op_tag.update, predication=predication, only_first=only_first)

    def find(self, predication, only_first=False) -> _T:
        entry = self
        if isinstance(predication, list):
            if len(predication) > 1:
                entry = self.extend(predication[:-1])
                predication = predication[-1]
            elif len(predication) == 1:
                predication = predication[0]

        return entry.pull(predication=predication, only_first=only_first)

    def erase(self):
        self.push(op=Entry.op_tag.erase)

    def iter(self):
        obj = self.pull()

        if isinstance(obj, Entry):
            yield from obj.iter()
        elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
            yield from obj
        else:
            raise NotImplementedError(obj)

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

    def _op_update(target, key, value):
        if not isinstance(value, collections.abc.Mapping):
            return Entry._op_assign(target, key, value)
        elif isinstance(target, (Entry, EntryContainer)):
            if key in (None, _undefined_, _not_found_):
                return target.update(value)
            else:
                return target.update({key: value})
        elif not isinstance(target, collections.abc.Mapping):
            raise TypeError(type(target))

        for k, v in value.items():
            if not isinstance(v, collections.abc.Mapping):
                target[k] = v
            else:
                tmp = target.setdefault(k, v)
                if tmp is v:
                    pass
                elif not isinstance(tmp, collections.abc.Mapping):
                    target[k] = v
                else:
                    Entry._op_update(tmp, None, v)

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
            return all([Entry._op_check(Entry._ht_get(target, Entry.normalize_query(k), _not_found_), v) for k, v in pred.items()])
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
    def _apply_filter(val, predication: collections.abc.Mapping, only_first=True):
        if not isinstance(val, (list, Entry, EntryContainer)):
            return val
        elif isinstance(predication, (int, slice)):
            return val[predication]
        elif not isinstance(predication, collections.abc.Mapping):
            return val

        def _filter(d):
            return all([Entry._ht_get(d, Entry.normalize_query(k), default_value=_not_found_) == v for k, v in predication.items()])

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
    def _ht_get(target, query, op: op_tag = _undefined_, default_value: _T = _undefined_,  predication=_undefined_, only_first=False) -> _T:
        """
            Finds an element with key equivalent to key.
            return if key exists return element else return default_value
        """
        if target is _not_found_ or target is None:
            return default_value

        if query is None:
            query = []
        elif not isinstance(query, list):
            query = [query]

        val = target
        last_idx = len(query)
        for idx, key in enumerate(query):
            val = _not_found_
            if key is None:
                val = target
            elif isinstance(target, (Entry, EntryContainer)):
                return target.get(query[idx:], default_value=default_value, op=op,
                                  predication=predication, only_first=only_first)
            elif isinstance(target, np.ndarray):
                try:
                    val = target[key]
                except (IndexError, KeyError, TypeError) as error:
                    logger.exception(error)
                    val = _not_found_
            elif isinstance(target, (collections.abc.Mapping)):
                val = target.get(key, _not_found_)
            elif isinstance(target, (collections.abc.Sequence)):
                if isinstance(key, int):
                    try:
                        val = target[key]
                    except (IndexError, KeyError, TypeError) as error:
                        logger.exception(error)
                        val = _not_found_
                else:
                    val = Entry._apply_filter(target, key)
                    if idx < last_idx and not only_first and isinstance(val, list):
                        return [Entry._ht_get(d,  query[idx+1:], default_value=default_value, op=op,
                                              predication=predication, only_first=only_first) for d in val]

            else:
                raise NotImplementedError(f"{type(target)} {type(key)} {query[:idx+1]}")

            if val is _not_found_:
                break
            target = val

        if predication is not _undefined_:
            val = Entry._apply_filter(val, predication=predication, only_first=only_first)

        if op is not _undefined_:
            val = Entry._apply_op(op, target=val)

        if val is not _not_found_:
            return val
        elif default_value is _undefined_:
            return Entry(target, query)
        else:
            return default_value

    @ staticmethod
    def _ht_put(target: Any, query: _TQuery, value: _T, create_if_not_exists=True, op=_undefined_,  predication=_undefined_, only_first=False) -> Tuple[_T, bool]:
        """
            insert if the key does not exist, does nothing if the key exists.
            return value at key
        """
        if not query:
            query = []

        if predication is _undefined_ and len(query) > 0:
            last_idx = len(query)-1
            target_key = query[-1]
        else:
            last_idx = len(query)
            target_key = None

        val = target
        idx = 0
        while idx < last_idx:
            key = query[idx]
            val = _not_found_

            if target is _not_found_ or target is None:
                break
            elif isinstance(target, (Entry,   EntryContainer)):
                val = target.put(query[idx:], value,
                                 op=op,
                                 create_if_not_exists=create_if_not_exists,
                                 predication=predication,
                                 only_first=only_first)
                idx = last_idx+1
            elif isinstance(target, (collections.abc.Mapping)):
                if not isinstance(key, str):
                    raise NotImplementedError(key)
                val = target.get(key, _not_found_)
                if val is not _not_found_:
                    pass
                elif not create_if_not_exists or idx == (len(query)-1):
                    break
                elif isinstance(query[idx+1], str):
                    val = _DICT_TYPE_()
                    target[key] = val
                else:
                    val = _LIST_TYPE_()
                    target[key] = val
            elif isinstance(target, (np.ndarray)):
                if not isinstance(key, (int, slice)):
                    raise TypeError(type(key))
                val = target[key]
            elif isinstance(target, collections.abc.Sequence):
                if isinstance(key, (int, slice)):
                    val = target[key]
                elif isinstance(key, (collections.abc.Sequence)):
                    val = [target[i] for i in key]
                elif isinstance(key, collections.abc.Mapping):
                    val = Entry._apply_filter(target, key, only_first=True)
                    if val is _not_found_:
                        val = deepcopy(key)
                        target.append(key)
                else:
                    raise TypeError(type(val))
            else:
                raise TypeError(f"{type(target)} {key}")

            target = val
            idx = idx+1

        if idx < last_idx:
            raise KeyError(query[idx:])
        elif idx > last_idx:
            return target
        elif isinstance(val, Entry.PRIMARY_TYPE):
            return target
        elif target_key is _next_:
            target.append(value)
            return target[-1]
        else:
            if predication is not _undefined_:
                target = Entry._apply_filter(target, predication=predication, only_first=only_first)
            if op is _undefined_:
                op = Entry.op_tag.assign

            if not only_first and isinstance(target, list):
                target = [Entry._apply_op(op, d, target_key, value) for d in target]
            else:
                target = Entry._apply_op(op, target, target_key, value)

            return target

    def pull(self, default_value=_undefined_, **kwargs) -> _T:
        return self._ht_get(self._cache, self._path, default_value=default_value,  **kwargs)

    def push(self,  value: _T = None,  **kwargs) -> Tuple[_T, bool]:
        if self._cache is None:
            if len(self._path) > 0 and isinstance(self._path[0], str):
                self._cache = _DICT_TYPE_()
            else:
                self._cache = _LIST_TYPE_()

        return self._ht_put(self._cache, self._path, value, **kwargs)

        # if isinstance(value, Entry.op_tag):
        #     value = {value: None}
        # else:
        #     v_entry: Entry = value if isinstance(value, Entry) else getattr(value, "_entry", _not_found_)

        #     # remove cycle reference
        #     if v_entry is not _not_found_ \
        #             and v_entry._cache is self._cache \
        #             and len(self._path) <= len(v_entry._path) \
        #             and all([v == v_entry._path[idx] for idx, v in enumerate(self._path)]):
        #         v_entry._cache = self._ht_get(self._cache, self._path, None)
        #         v_entry._path = v_entry._path[len(self._path):]
        #         logger.debug((v_entry._cache, self._path, v_entry._path))

    def has_child(self, member_name) -> bool:
        return self._ht_get(self._cache, self._path+[member_name], op=Entry.op_tag.exists)

    def flush(self, v):
        self._cache = self.push(v, op=Entry.op_tag.try_insert)
        self._path = []


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
            val = [Entry._ht_get(d, self._path, default_value=_not_found_, **kwargs) for d in self._d_list]

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
