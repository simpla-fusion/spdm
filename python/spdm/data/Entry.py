import collections
import collections.abc
import dataclasses
import functools
import inspect
import operator
from copy import deepcopy
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

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
    elif value is None:  # and target._cache is None:
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


def ht_find(target,  query: Optional[_TQuery] = None, /,  default_value=_undefined_, only_first=False) -> Any:
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
            val = Entry(target, path=query[idx:])
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
            if only_first is True:
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
        return Entry(target, path=query[idx:])
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
    return v in ht_find(target,  *args,  **kwargs)


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


_TEntry = TypeVar('_TEntry', bound='Entry')


class Entry(object):
    is_entry = True
    __slots__ = "_cache", "_path"

    def __init__(self, cache=None,  *args, path=None,      **kwargs):
        super().__init__()
        self._cache = None
        self._path = normalize_query(path)
        self.enable_cache(cache)

    def duplicate(self) -> _TEntry:
        return self.__class__(self._cache, path=self._path)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} root={type(self._cache)} path={self._path} />"

    @property
    def writable(self) -> bool:
        return True

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

    def child(self, rquery: _TQuery) -> _TEntry:
        if rquery is None:
            return self
        else:
            node = self.duplicate()
            node._path = node._path + normalize_query(rquery)
            return node

    @property
    def parent(self) -> _TEntry:
        if not self._path:
            raise RuntimeError(f"No parent")
        node = self.duplicate()
        node._path = node._path[:-1]
        return node

    @property
    def siblings(self) -> _TEntry:
        """
            return next brother neighbour
        """
        return self.parent.child({"@not": self._path[-1]})

    def __next__(self) -> _TEntry:
        return self.parent.child({"@next": self._path[-1]})

    def reset(self, value=None) -> _TEntry:
        self._cache = value
        self._path = []
        return self

    @property
    def exists(self) -> bool:
        if not self._path:
            return self._cache is not None
        else:
            return ht_find(self._cache, self._path, default_value=_not_found_) is not _not_found_

    @property
    def is_list(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Sequence) and not isinstance(self._cache, str)

    @property
    def is_dict(self) -> bool:
        return not self._path and isinstance(self._cache, collections.abc.Mapping)

    @property
    def count(self) -> int:
        if self._cache is None:
            return 0
        elif not self._path:
            return len(self._cache)
        else:
            return ht_count(self._cache,  self._path)

    def get(self, /, default_value: _T = _undefined_, only_first=_undefined_, **kwargs) -> Union[_T, _TEntry]:
        if not self._path:
            return self._cache
        else:
            value = ht_find(self._cache,  self._path, default_value=_not_found_, only_first=only_first, **kwargs)

            if isinstance(value, (int, float, str, np.ndarray)):
                return value
            elif isinstance(value, (collections.abc.Sequence)):
                if isinstance(self._path[-1], collections.abc.Mapping):
                    if only_first is not False and len(value) == 1:
                        value = value[0]
                return Entry(value)
            elif isinstance(value,  collections.abc.Mapping):
                return Entry(value)
            elif value is not _not_found_:
                return value
            elif default_value is not _undefined_:
                return default_value
            else:
                return self

    def put(self,  value, /, assign_if_exists=True):
        if not self._path:
            if assign_if_exists or self._cache is None:
                self._cache = value
            return self._cache
        else:
            if self._cache is None:
                if isinstance(self._path[0], str):
                    self._cache = _DICT_TYPE_()
                else:
                    self._cache = _LIST_TYPE_()

            return ht_insert(self._cache,  self._path, value, assign_if_exists=assign_if_exists)

    def find(self, query: _TQuery, default_value=_not_found_,  /, **kwargs) -> Any:
        return self.child(query).get(default_value=default_value, **kwargs)

    def insert(self, query: _TQuery, value: _T, /, **kwargs) -> _T:
        return self.child(query).put(value, **kwargs)

    def append(self, value) -> _TEntry:
        target = self.put(_LIST_TYPE_(), assign_if_exists=False)

        if not isinstance(target, collections.abc.Sequence):
            raise ValueError(type(target))

        target.append(value)
        return Entry(target, path=[len(target)-1])

    def update(self, value=None, /, **kwargs) -> None:
        if not self._path and self._cache is None:
            self._cache = value
            return value
        else:
            if self._cache is None or self._cache is _not_found_:
                if isinstance(self._path[0], str):
                    self._cache = _DICT_TYPE_()
                else:
                    self._cache = _LIST_TYPE_()
            return ht_update(self._cache,  self._path, value, **kwargs)

    def equal(self, other, /, **kwargs) -> bool:
        if not self._path:
            return self._cache == other
        else:
            val = self.get()
            if isinstance(val, Entry):
                return val.equal(other)
            else:
                return val == other

    def update_many(self, value=None, /, **kwargs):
        raise NotImplementedError()

    def erase(self,   /, **kwargs) -> bool:
        if not self._path:
            self._cache = None
            return True
        else:
            return ht_erase(self._cache,  self._path,   **kwargs)

    def call(self,   *args, **kwargs) -> Any:
        obj = self.fetch(default_value=_not_found_)
        if callable(obj):
            res = obj(*args, **kwargs)
        elif len(args)+len(kwargs) == 0:
            res = obj
        else:
            raise TypeError(f"'{type(obj)}' is not callable")
        return res

    def iter(self,  **kwargs):
        obj = self.fetch(**kwargs)

        if isinstance(obj, Entry):
            yield from obj.iter()
        elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
            yield from obj
        else:
            raise NotImplementedError(type(obj))

    def items(self,  **kwargs):
        yield from ht_items(self._cache, self._path,  **kwargs)

    def values(self,  **kwargs):
        yield from ht_values(self._cache, self._path,   **kwargs)

    def keys(self,    **kwargs):
        yield from ht_keys(self._cache, self._path,  **kwargs)

    def __serialize__(self, *args, **kwargs):
        return [v for v in self.values(*args, **kwargs)]


class EntryWrapper(Entry):

    def __init__(self,  *sources, **kwargs):
        super().__init__(sources, **kwargs)

    def find(self,  *args, default_value=_undefined_, **kwargs):
        res = next(filter(lambda d: d is not _not_found_, map(
            lambda d: ht_find(d, self._path, default_value=_not_found_), self._cache)), default_value)
        if res is _undefined_:
            res = EntryWrapper(self._cache, path=self._path)
        return res

    def insert(self,    value, *args, **kwargs):
        return ht_insert(self._cache[0], self._path, value, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.find())

    def __contains__(self, v) -> bool:
        return next(filter(lambda d: ht_contains(d, v), self._cache), False)

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

    #  def get(self, query=[], *args, default_value=_not_found_, **kwargs):
    #     query = self._path + normalize_query(query)
    #     obj = self._cache
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
    #     query = self._path + normalize_query(query)
    #     obj = self._cache
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
    def __init__(self, d_list: Sequence, *args,   cache=False, reducer=None, **kwargs):
        super().__init__(cache, *args,  **kwargs)
        if not isinstance(d_list, collections.abc.Sequence):
            raise TypeError(type(d_list))
        self._d_list = d_list
        self._reducer = reducer if reducer is not None else operator.__add__

    def duplicate(self):
        return self.__class__(self._d_list, cache=self._cache, reducer=self._reducer, path=self._path)

    def get(self,  *args, default_value=_not_found_, cache: str = "on",  **kwargs) -> Any:
        res = _not_found_

        if self._cache is not None and cache not in ("off", "no"):
            res = super().get(default_value=_not_found_,  **kwargs)
            if res is not _not_found_ or cache == "only":
                return res

        query = [slice(None, None, None)] + self._path

        cache = ht_find(self._d_list, query, *args, default_value=_not_found_, **kwargs)

        if isinstance(cache, collections.abc.Sequence):
            cache = [d if not isinstance(d, Entry) else d.get()
                     for d in cache if (d is not None and d is not _not_found_)]

        if isinstance(cache, collections.abc.Sequence) and len(cache) > 0:
            res = functools.reduce(self._reducer, cache[1:], cache[0])
            if self._cache is not None:
                super().put(res)
        else:
            res = default_value

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
