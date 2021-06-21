import collections
import collections.abc
import dataclasses
import functools
import operator
import inspect
from typing import (Any, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)
from copy import deepcopy
from ..numlib import np
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, serialize

_next_ = object()
_last_ = object()

_TObject = TypeVar("_TObject")
_TPath = TypeVar("_TPath", int, slice, str,  Sequence)
_TQuery = TypeVar("_TQuery", int,  slice, str, Sequence, Mapping)

_TKey = TypeVar('_TKey', int, str)
_TIndex = TypeVar('_TIndex', int, slice)
_DICT_TYPE_ = dict
_LIST_TYPE_ = list


def normalize_query(query):
    if query is None:
        query = []
    elif isinstance(query, str):
        # TODO: parse uri request
        query = query.split('.')
    elif isinstance(query, tuple):
        query = list(query)
    elif not isinstance(query, collections.abc.MutableSequence):
        query = [query]
    return query


def ht_insert(target: Any, query: _TQuery,  value: _TObject, assign_if_exists=False,  **kwargs) -> _TObject:
    """
        insert if the key does not exist, does nothing if the key exists.
        return value at key
    """
    if isinstance(target, Entry):
        return target.insert(query, value, assign_if_exists=assign_if_exists, **kwargs)

    query = normalize_query(query)

    if len(query) > 0:
        pass
    elif value is None:  # and target._data is None:
        return target
    else:
        raise RuntimeError(f"Empty query! {type(target)} {type(value)}")

    val = _not_found_
    for idx, key in enumerate(query):
        if isinstance(target, Entry):
            target = target.insert(query[idx:], value, assign_if_exists=assign_if_exists, **kwargs)
            break

        if idx == len(query)-1:
            child = value
        else:
            child = _DICT_TYPE_() if isinstance(query[idx+1], str) else _LIST_TYPE_()

        if key is _next_ or (isinstance(target, collections.abc.MutableSequence) and key == len(target)):
            if not isinstance(target, collections.abc.MutableSequence):
                raise TypeError(type(target))
            target.append(child)
            val = target[-1]
        elif isinstance(key,  int):
            if not isinstance(target, (collections.abc.MutableSequence, np.ndarray)):
                raise TypeError(type(target))
            elif key >= len(target):
                raise IndexError(f"Out of range! {key}>={len(target)}")
            elif assign_if_exists:
                target[key] = child
            val = target[key]
        elif isinstance(key,  slice):
            if not isinstance(target, (np.ndarray)):
                for idx in range(key.start, key.stop, key.step):
                    ht_insert(target, [idx]+query[idx+1:], value, assign_if_exists=assign_if_exists, **kwargs)
                val = ht_find(target, key)
                break
            elif assign_if_exists:
                target[key] = child
            val = target[key]
        elif isinstance(key, str):
            if (assign_if_exists and idx == (len(query)-1)):
                target[key] = child
                val = target[key]
            else:
                val = target.get(key, _not_found_)
                if val is _not_found_:
                    target[key] = child
                    val = target[key]

            if val is _not_found_:
                try:
                    val = target.find(key, default_value=_not_found_)
                except Exception:
                    val = _not_found_
        elif isinstance(key, collections.abc.Mapping):
            val = ht_find(target, key, default_value=_not_found_)

            if val is _not_found_:
                # FIXME: !!!! not complete !!!!
                val = ht_insert(target, _next_, deepcopy(key))

        if val is _not_found_:
            break
        else:
            target = val

    if val is _not_found_:
        raise KeyError(query[idx:])
    return val


def ht_update(target,  query: Optional[_TQuery], value, /,  **kwargs) -> Any:
    if query is not None and len(query) > 0:
        val = ht_insert(target, query, _not_found_,   **kwargs)
    else:
        val = target

    if isinstance(val, Entry):
        val.update(None, value,   **kwargs)
    elif isinstance(val, dict):
        for k, v in value.items():
            u = val.setdefault(k, v)
            if u is not v:
                ht_update(u, None, v,  **kwargs)

    elif hasattr(val, '_entry'):
        logger.debug(val.__class__)
        val.update(value, **kwargs)
    else:
        ht_insert(target, query, value, assign_if_exists=True, **kwargs)


def ht_find(target,  query: Optional[_TQuery] = None, /,  default_value=_undefined_, only_first=True) -> Any:
    """
        Finds an element with key equivalent to key.
        return if key exists return element else return default_value
    """
    if isinstance(target, Entry):
        return target.find(query, default_value=default_value)
    elif target is _not_found_:
        return default_value

    query = normalize_query(query)

    val = target

    for idx, key in enumerate(query):
        if target is None or target is _not_found_:
            val = target
            break
        elif isinstance(target, Entry):
            val = target.find(query[idx:], default_value)
            break
        elif key is _next_ or (isinstance(target, collections.abc.Sequence) and key == len(target)):
            val = Entry(target, prefix=query[idx:])
            break
        elif isinstance(key, str):
            if hasattr(target.__class__, 'get'):
                val = target.get(key, _not_found_)
            else:
                try:
                    val = target[key]
                except KeyError as error:
                    logger.debug(f"Can not index {type(target)} by {key}! \nError: {error}")
                    val = _not_found_
        elif isinstance(key, int):
            try:
                val = target[key]
            except IndexError as error:
                logger.debug(f"Can not index {type(target)} by {key}! \nError: {error}")
                val = _not_found_
        elif isinstance(key, slice) and isinstance(target, np.ndarray):
            val = target[key]
        elif isinstance(key, slice):
            if (isinstance(target, collections.abc.Sequence) and not isinstance(target, str)):
                val = [ht_find(v, query[idx+1:], default_value=_not_found_) for v in val[key]]
            elif isinstance(target, collections.abc.Mapping):
                raise NotImplementedError(type(target))
            else:
                length = ht_count(target)
                start = key.start if key.start is not None else 0
                if key.stop is None:
                    stop = length
                elif key.stop < 0:
                    stop = (key.stop+length) % length
                else:
                    stop = key.stop

                step = key.step

                val = [ht_find(target, [s]+query[idx:], default_value=_not_found_) for s in range(start, stop, step)]

            break
        elif isinstance(key, collections.abc.Mapping):
            if key.get("_only_first", only_first):
                try:
                    val = next(filter(lambda d: ht_check(d, key), ht_iter(target)))
                except StopIteration:
                    val = _not_found_
            else:
                val = [d for d in ht_iter(target) if ht_check(d, key)]
                if len(val) == 0:
                    val = _not_found_
        else:
            raise TypeError(type(key))

        if val is _not_found_ or val is None:
            break
        else:
            target = val

    if not (val is _not_found_ or val is None):
        return val
    elif default_value is _undefined_:
        return Entry(target, prefix=query[idx:])
    else:
        return default_value

    # elif isinstance(obj, collections.abc.Mapping):
    #     if not isinstance(key, str):
    #         raise TypeError(f"mapping indices must be str, not {type(key).__name__}! \"{query}\"")
    #     tmp = obj.find(key, _not_found_)
    #     obj = tmp
    # elif isinstance(obj, collections.abc.MutableSequence):
    #     if not isinstance(key, (int, slice)):
    #         raise TypeError(
    #             f"list indices must be integers or slices, not {type(key).__name__}! \"{query[:idx+1]}\" {type(obj)}")
    #     elif isinstance(key, int) and isinstance(target._data, collections.abc.MutableSequence) and key > len(target._data):
    #         raise IndexError(f"Out of range! {key} > {len(target._data)}")
    #     obj = obj[key]
    # elif hasattr(obj, "_cache"):
    #     if obj._cache._data == target._data and obj._cache._prefix == query[:idx]:
    #         suffix = query
    #         obj = obj._cache._data
    #         break
    #     else:
    #         obj = obj._cache.find(query[idx:])
    #     break

    # if rquery is None:
    #     target._data = obj
    #     target._prefix = []


def ht_check(target, condition: Mapping) -> bool:
    def _check_eq(l, r) -> bool:
        if l is r:
            return True
        elif type(l) is not type(r):
            return False
        elif isinstance(l, np.ndarray):
            return np.allclose(l, r)
        else:
            return l == r
    d = [_check_eq(ht_find(target, k, default_value=_not_found_), v)
         for k, v in condition.items() if not isinstance(k, str) or k[0] != '_']
    return all(d)


def ht_erase(target, query: Optional[_TQuery] = None, *args,  **kwargs):

    if isinstance(target, Entry):
        return target.remove(query, *args, **kwargs)

    query = normalize_query(query)

    if len(query) == 0:
        return False

    target = ht_find(target, query[:-1], _not_found_)

    if target is _not_found_:
        return
    elif isinstance(query[-1], str):
        try:
            delattr(target, query[-1])
        except Exception:
            try:
                del target[query[-1]]
            except Exception:
                raise KeyError(f"Can not delete '{query}'")


def ht_count(target,    *args, default_value=_not_found_, **kwargs) -> int:
    if isinstance(target, Entry):
        return target.count(*args, **kwargs)
    else:
        target = ht_find(target, *args, default_value=default_value, **kwargs)
        if target is None or target is _not_found_:
            return 0
        elif isinstance(target, (str, int, float, np.ndarray)):
            return 1
        elif isinstance(target, (collections.abc.Sequence, collections.abc.Mapping)):
            return len(target)
        else:
            raise TypeError(f"Not countable! {type(target)}")


def ht_contains(target, v,  *args,  **kwargs) -> bool:
    return v in target.find(*args, **kwargs)


def ht_iter(target, query=None, /,  **kwargs):
    target = ht_find(target, query, default_value=_not_found_)
    if target is _not_found_:
        yield from []
    elif isinstance(target, (int, float, np.ndarray)):
        yield target
    elif isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
        yield from target
    elif isinstance(target, Entry):
        yield from target.iter()
    else:
        yield target


def ht_items(target, query: Optional[_TQuery], *args, **kwargs):
    obj = ht_find(target, query, *args, **kwargs)
    if ht_count(obj) == 0:
        yield from {}
    elif isinstance(obj, Entry):
        yield from obj.items()
    elif isinstance(obj, collections.abc.Mapping):
        yield from obj.items()
    elif isinstance(obj, collections.abc.MutableSequence):
        yield from enumerate(obj)
    elif isinstance(obj, Entry):
        yield from obj.items()
    else:
        raise TypeError(type(obj))


def ht_values(target, query: _TQuery = None, /, **kwargs):
    target = ht_find(target, query, **kwargs)
    if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
        yield from target
    elif isinstance(target, collections.abc.Mapping):
        yield from target.values()
    elif isinstance(target, Entry):
        yield from target.iter()
    else:
        yield target


def ht_keys(target, query: _TQuery = None, /, **kwargs):
    target = ht_find(target, query, **kwargs)
    if isinstance(target, collections.abc.Mapping):
        yield from target.keys()
    elif isinstance(target, collections.abc.MutableSequence):
        yield from range(len(target))
    else:
        raise NotImplementedError()


def ht_compare(first, second) -> bool:
    if isinstance(first, Entry):
        first = first.find()
    if isinstance(second, Entry):
        second = second.find()
    return first == second


class Entry(object):
    is_entry = True
    __slots__ = "_data", "_prefix"

    def __init__(self, data=None,  *args, prefix=None,      **kwargs):
        super().__init__()
        self._data = data
        self._prefix = normalize_query(prefix)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} data={type(self._data)} prefix={self._prefix} />"

    @property
    def writable(self) -> bool:
        return True

    @property
    def data(self):
        return self._data

    @property
    def parent(self):
        return self._parent

    @property
    def prefix(self):
        return self._prefix

    @property
    def is_relative(self):
        return len(self._prefix) > 0

    def __eq__(self, o: Any) -> bool:
        return isinstance(o, Entry) and self._data is o._data and self._prefix == o._prefix

    @property
    def empty(self):
        return (self._data is None and len(self._prefix) == 0) or self.find(None, default_value=None) is None

    @property
    def invalid(self):
        return self._data is None

    def append(self, query):
        self._prefix += normalize_query(query)

    def extend(self, query):
        return self.__class__(self._data, prefix=self._prefix + normalize_query(query))

    def child(self, query, *args, **kwargs):
        if not query:
            return self
        else:
            return self.__class__(self._data, prefix=self._prefix + normalize_query(query))

    def copy(self, other):
        raise NotImplementedError()

    # def get(self, *args, **kwargs) -> Any:
    #     return self.find(*args, **kwargs)

    # def put(self,  rquery:  Optional[_TQuery], value) -> Any:
        return self.insert(rquery, value, assign_if_exists=True)

    def find(self, rquery: Optional[_TQuery] = None, /, default_value=_undefined_,  **kwargs) -> Any:
        return ht_find(self._data,  self._prefix + normalize_query(rquery), default_value=default_value, **kwargs)

    def get(self, rquery: Optional[_TQuery] = None, /, default_value=_undefined_,  **kwargs) -> Any:
        """ alias of find """
        return self.find(rquery, default_value=default_value, **kwargs)

    def _before_insert(self, query):
        if isinstance(query[0], str):
            self._data = _DICT_TYPE_()
        else:
            self._data = _LIST_TYPE_()

    def insert(self, rquery: Optional[_TQuery], value, /, **kwargs):
        if hasattr(value, '_entry') and self.extend(rquery) == value._entry:
            # break cycle reference
            value._entry = Entry(value._entry.find(default_value=None))

        query = self._prefix + normalize_query(rquery)

        if len(query) == 0:
            self._data = value
            return value
        else:
            if self._data is None or self._data is _not_found_:
                if isinstance(query[0], str):
                    self._data = _DICT_TYPE_()
                else:
                    self._data = _LIST_TYPE_()

            return ht_insert(self._data,  query, value,  **kwargs)

    def update(self, rquery: Optional[_TQuery] = None,   value=None, /, reset=False, **kwargs):
        if rquery is None and reset is True:
            self._data = value
            return
        elif value is None:
            return

        query = self._prefix + normalize_query(rquery)
        if len(query) == 0 and self._data is None:
            self._data = value
        else:
            if self._data is None or self._data is _not_found_:
                if isinstance(query[0], str):
                    self._data = _DICT_TYPE_()
                else:
                    self._data = _LIST_TYPE_()
            ht_update(self._data, query, value, **kwargs)

    def update_many(self, rquery: Optional[_TQuery],   value=None, /, **kwargs):
        raise NotImplementedError()

    def erase(self, rquery: Optional[_TQuery] = None, /, **kwargs) -> None:
        return ht_erase(self._data,  self._prefix + normalize_query(rquery),   **kwargs)

    def count(self,  rquery: Optional[_TQuery] = None,  /, **kwargs) -> int:
        return ht_count(self._data,  self._prefix + normalize_query(rquery),    **kwargs)

    def contains(self, v, rquery=None,  /, **kwargs) -> bool:
        return ht_contains(self._data,  self._prefix + normalize_query(rquery),    **kwargs)

    def call(self,   rquery: Optional[_TQuery], *args, **kwargs) -> Any:
        obj = self.find(rquery, _not_found_)
        if callable(obj):
            res = obj(*args, **kwargs)
        elif len(args)+len(kwargs) == 0:
            res = obj
        else:
            raise TypeError(f"'{type(obj)}' is not callable")
        return res

    def compare(self, other) -> bool:
        return ht_compare(self.find(), other)

    def iter(self, *args, **kwargs):
        obj = self.find(*args, **kwargs)

        if isinstance(obj, Entry):
            yield from obj.iter()
        elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
            yield from obj
        else:
            raise NotImplementedError(type(obj))

    def items(self, query: Optional[_TQuery] = None, * args, **kwargs):
        yield from ht_items(self._data, self._prefix+normalize_query(query), *args, **kwargs)

    def values(self,  query: Optional[_TQuery] = None,  *args, **kwargs):
        yield from ht_values(self._data, self._prefix+normalize_query(query), *args, **kwargs)

    def keys(self,  query: Optional[_TQuery] = None, *args, **kwargs):
        yield from ht_keys(self._data, self._prefix+normalize_query(query), *args, **kwargs)

    def __serialize__(self, *args, **kwargs):
        return [v for v in self.values(*args, **kwargs)]


class EntryWrapper(Entry):

    def __init__(self,  *sources, **kwargs):
        super().__init__(sources, **kwargs)

    def find(self, rquery: Optional[_TQuery], *args, default_value=_undefined_, **kwargs):
        query = self._prefix+normalize_query(rquery)
        res = next(filter(lambda d: d is not _not_found_, map(
            lambda d: ht_find(d, query, default_value=_not_found_), self._data)), default_value)
        if res is _undefined_:
            res = EntryWrapper(self._data, prefix=query)
        return res

    def insert(self,  rquery: Optional[_TQuery], value, *args, **kwargs):
        return ht_insert(self._data[0], self._prefix+normalize_query(rquery), value, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.find())

    def __contains__(self, v) -> bool:
        return next(filter(lambda d: ht_contains(d, v), self._data), False)

    def iter(self):
        for d in self._data:
            yield from d.iter()

    def items(self):
        for k in self.keys():
            yield k, self.find(k)

    def keys(self):
        k = set()
        for d in self._data:
            k.add(d.keys())
        yield from k

    def values(self):
        for k in self.keys():
            yield self.find(k)

    #  def get(self, query=[], *args, default_value=_not_found_, **kwargs):
    #     query = self._prefix + normalize_query(query)
    #     obj = self._data
    #     if obj is None:
    #         obj = self._parent
    #     for p in query:
    #         if type(p) is str and hasattr(obj, p):
    #             obj = getattr(obj, p, _not_found_)
    #         elif obj is not None:
    #             try:
    #                 obj = obj[p]
    #             except IndexError:
    #                 obj = _not_found_
    #             except TypeError:
    #                 obj = _not_found_
    #         else:
    #             raise KeyError(query)
    #     return obj

    # def put(self,  query, value, *args, **kwargs):
    #     query = self._prefix + normalize_query(query)
    #     obj = self._data
    #     if len(query) == 0:
    #         return obj
    #     for p in query[:-1]:
    #         if type(p) is str and hasattr(obj, p):
    #             obj = getattr(obj, p)
    #         else:
    #             try:
    #                 t = obj[p]
    #             except KeyError:
    #                 obj[p] = {}
    #                 obj = obj[p]
    #             except IndexError as error:
    #                 raise IndexError(f"{p} > {len(obj)}! {error}")
    #             else:
    #                 obj = t
    #         # elif type(p) is int and p < len(obj):
    #         #     obj = obj[p]
    #         # else:
    #         #     obj[p] = {}
    #         #     obj = obj[p]
    #     if hasattr(obj, query[-1]):
    #         setattr(obj, query[-1], value)
    #     else:
    #         obj[query[-1]] = value

    #     return obj[query[-1]]


class EntryCombiner(Entry):
    def __init__(self, data: Union[Entry, Sequence], *args, prefix=None, reducer=None, **kwargs):
        prefix = normalize_query(prefix)
        if isinstance(data, Entry):
            prefix = data._prefix+prefix
            data = data._data
        if not isinstance(data, collections.abc.Sequence):
            raise TypeError(type(data))

        super().__init__(data, *args, prefix=prefix, **kwargs)

        self._reducer = reducer if reducer is not None else operator.__add__

    @property
    def writable(self) -> bool:
        return False

    def find(self, rquery: Optional[_TQuery] = None, *args, default_value=None,   **kwargs) -> Any:

        cache = ht_find(self._data, [slice(None, None, None)] + self._prefix +
                        normalize_query(rquery), *args, default_value=_not_found_, **kwargs)

        if cache is _not_found_:
            return default_value
        elif isinstance(cache, collections.abc.Sequence):
            cache = [d for d in cache if (d is not None and d is not _not_found_)]
            if len(cache) == 0:
                return default_value

        if all([isinstance(d, (Entry, collections.abc.Mapping, collections.abc.Sequence)) or hasattr(d, "_entry") for d in cache]):
            return EntryCombiner(cache, reducer=self._reducer)
        else:
            try:
                data = [d for d in cache
                        if not (isinstance(d, Entry) or hasattr(d, '_entry'))]

                res = functools.reduce(self._reducer, data[1:], data[0])
            except Exception as error:
                raise error
            return res

    def insert(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def erase(self, *args, **kwargs):
        raise NotImplementedError()


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
