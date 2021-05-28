
import collections.abc
from typing import (Any, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from ..numlib import np
from ..util.logger import logger
from ..util.utilities import (_not_defined_, _not_found_, normalize_path,
                              serialize)

_next_ = object()
_last_ = object()

_TObject = TypeVar("_TObject")
_TPath = TypeVar("_TPath", str, float, slice, Sequence)
_TKey = TypeVar('_TKey', int, str)
_TIndex = TypeVar('_TIndex', int, slice)
_DICT_TYPE_ = dict
_LIST_TYPE_ = list


def ht_insert(target: Any, path: _TPath,  value: _TObject, assign_if_exists=False, ignore_attribute=True) -> _TObject:
    """
        insert if the key does not exist, does nothing if the key exists.
        return value at key
    """
    if isinstance(target, Entry):
        return target.insert(path, value, assign_if_exists=assign_if_exists)

    path = normalize_path(path)

    if len(path) > 0:  # and target._data is None:
        pass
    elif hasattr(target, "_cache"):
        target._cache = value
        return target
    else:
        raise RuntimeError(f"Empty path!")
    val = _not_found_
    for idx, key in enumerate(path):
        if isinstance(target, Entry):
            target = target.insert(path[idx:], value, assign_if_exists=assign_if_exists)
            break

        if idx == len(path)-1:
            child = value
        else:
            child = _DICT_TYPE_() if isinstance(path[idx+1], str) else _LIST_TYPE_()

        if key is _next_ or (isinstance(target, collections.abc.MutableSequence) and key == len(target)):
            if not isinstance(target, collections.abc.MutableSequence):
                raise TypeError(type(target))
            target.append(child)
            val = target[-1]
        elif isinstance(key,  int):
            if not isinstance(target, collections.abc.MutableSequence):
                raise TypeError(type(target))
            elif assign_if_exists:
                target[key] = child
            val = target[key]
        elif isinstance(key,  slice):
            for idx in range(key.start, key.stop, key.step):
                ht_insert(target, [idx]+path[idx+1:], value, assign_if_exists=assign_if_exists)
            val = ht_get(target, key)
            break
        elif isinstance(key, str):
            if not ignore_attribute:
                val = getattr(target, key, _not_found_)
                if val is _not_found_ and assign_if_exists:
                    try:
                        setattr(target, key, child)
                        val = getattr(target, key, _not_found_)
                    except AttributeError:
                        val = _not_found_

            if ignore_attribute or val is _not_found_:
                try:
                    if assign_if_exists:
                        target[key] = child
                        val = target[key]
                    else:
                        val = target.setdefault(key, child)
                except Exception as error:
                    logger.debug(error)
                    val = _not_found_
            if val is _not_found_:
                try:
                    val = target.get(key, _not_found_)
                except Exception:
                    val = _not_found_

        if val is _not_found_:
            break
        else:
            target = val

    if val is _not_found_:
        raise KeyError(path[idx:])
    return val


def ht_get(target,  path: Optional[_TPath] = None, default_value=_not_defined_, ignore_attribute=True) -> Any:
    """
        Finds an element with key equivalent to key.
        return if key exists return element else return default_value
    """
    if isinstance(target, Entry):
        return target.get(path, default_value=default_value)

    path = normalize_path(path)

    val = target
    for idx, key in enumerate(path):
        if isinstance(target, Entry):
            val = target.get(path[idx:], default_value)
            break
        elif key is _next_ or (isinstance(target, collections.abc.Sequence) and key == len(target)):
            val = Entry(target, prefix=path[idx:])
            break
        elif isinstance(key, str):
            if not ignore_attribute:
                try:
                    val = getattr(target, key, _not_found_)
                except Exception:
                    val = _not_found_

            if ignore_attribute or val is _not_found_:
                try:
                    val = target[key]
                except Exception:
                    val = _not_found_
        elif isinstance(key,  (int, slice)):
            try:
                val = target[key]
            except Exception:
                val = _not_found_

        if val is _not_found_:
            break
        else:
            target = val

    if val is not _not_found_:
        return val
    elif default_value is _not_defined_:
        return Entry(target, prefix=path[idx:])
    else:
        return default_value

    # elif isinstance(obj, collections.abc.Mapping):
    #     if not isinstance(key, str):
    #         raise TypeError(f"mapping indices must be str, not {type(key).__name__}! \"{path}\"")
    #     tmp = obj.get(key, _not_found_)
    #     obj = tmp
    # elif isinstance(obj, collections.abc.MutableSequence):
    #     if not isinstance(key, (int, slice)):
    #         raise TypeError(
    #             f"list indices must be integers or slices, not {type(key).__name__}! \"{path[:idx+1]}\" {type(obj)}")
    #     elif isinstance(key, int) and isinstance(target._data, collections.abc.MutableSequence) and key > len(target._data):
    #         raise IndexError(f"Out of range! {key} > {len(target._data)}")
    #     obj = obj[key]
    # elif hasattr(obj, "_cache"):
    #     if obj._cache._data == target._data and obj._cache._prefix == path[:idx]:
    #         suffix = path
    #         obj = obj._cache._data
    #         break
    #     else:
    #         obj = obj._cache.get(path[idx:])
    #     break

    # if rpath is None:
    #     target._data = obj
    #     target._prefix = []


def ht_update(target,  value, rpath: Optional[_TPath] = None, *args, ignore_attribute=True, **kwargs):
    if isinstance(target, Entry):
        if target.writable:
            target.p
    if not isinstance(value, collections.abc.Mapping):
        target.put(value, rpath)
    # elif rpath is None:
    #     target.put(value)
    else:
        obj = ht_get(target, rpath, _not_found_)
        if isinstance(obj, collections.abc.MutableMapping):
            for k, v in value.items():
                target = ht_insert(obj, k, v)
                if target is v:
                    pass
                elif hasattr(target.__class__, 'update'):
                    target.update(v)
                else:
                    raise KeyError

            # obj.update(value)
        else:
            ht_insert(target, rpath, value, assign_if_exists=True)
    return target


def ht_erase(target, path: Optional[_TPath] = None, *args, ignore_attribute=True, **kwargs):

    if isinstance(target, Entry):
        return target.remove(path, *args, **kwargs)

    path = normalize_path(path)

    if len(path) == 0:
        return False

    target = ht_get(target, path[:-1], _not_found_)

    if target is _not_found_:
        return
    elif isinstance(path[-1], str):
        try:
            delattr(target, path[-1])
        except Exception:
            try:
                del target[path[-1]]
            except Exception:
                raise KeyError(f"Can not delete '{path}'")


def ht_count(target,    *args, default_value=_not_found_, **kwargs) -> int:
    if isinstance(target, Entry):
        return target.count(*args, **kwargs)
    else:
        target = ht_get(target, *args, default_value=default_value, **kwargs)
        if target is None or target is _not_found_:
            return 0
        elif isinstance(target, (str, int, float, np.ndarray)):
            return 1
        elif isinstance(target, (collections.abc.Sequence, collections.abc.Mapping)):
            return len(target)
        else:
            raise TypeError(f"Not countable! {type(target)}")


def ht_contains(target, v,  *args, ignore_attribute=True, **kwargs) -> bool:
    return v in target.get(*args, **kwargs)


def ht_iter(target, *args, default_value=None, **kwargs):
    obj = target.get(*args, default_value=default_value if default_value is not None else [], **kwargs)

    if isinstance(obj, Entry):
        yield from obj.iter()
    elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
        yield from obj
    else:
        raise NotImplementedError(type(obj))


def ht_items(target,  *args, **kwargs):
    obj = target.get(*args, **kwargs)
    if isinstance(obj, collections.abc.Mapping):
        yield from obj.items()
    elif isinstance(obj, collections.abc.MutableSequence):
        yield from enumerate(obj)
    elif isinstance(obj, Entry):
        yield from obj.items()
    else:
        raise TypeError(type(obj))


def ht_values(target,  *args, **kwargs):
    obj = target.get(*args, **kwargs)
    if isinstance(obj, collections.abc.Mapping):
        yield from obj.values()
    elif isinstance(obj, collections.abc.MutableSequence):
        yield from obj
    elif isinstance(obj, Entry):
        yield from []
    else:
        yield obj


def ht_keys(target,  *args, **kwargs):
    obj = target.get(*args, **kwargs)
    if isinstance(obj, collections.abc.Mapping):
        yield from obj.keys()
    elif isinstance(obj, collections.abc.MutableSequence):
        yield from range(len(obj))
    else:
        raise NotImplementedError()


def ht_compare(first, second) -> bool:
    if isinstance(first, Entry):
        first = first.fetch()
    if isinstance(second, Entry):
        second = second.fetch()
    return first == second


class Entry(object):
    is_entry = True
    __slots__ = "_data", "_prefix"

    def __init__(self, data=_not_found_,  *args, prefix=None,      **kwargs):
        super().__init__()
        self._data = data
        self._prefix = normalize_path(prefix)

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

    @property
    def empty(self):
        return (self._data is None and len(self._prefix) == 0) or self.get(default_value=_not_found_) is _not_found_

    def fetch(self):
        return self.get(default_value=_not_found_)

    def resolve(self):
        if self._prefix is None:
            return self

        data = self.get(default_value=_not_found_)

        if isinstance(data, Entry):
            return data
        else:
            return Entry(data, prefix=[])

    def append(self, path):
        self._prefix += normalize_path(path)

    def extend(self, path):
        res = self.__class__(self._data, prefix=self._prefix, parent=self._parent)
        res.append(path)
        return res

    def child(self, path, *args, **kwargs):
        if not path:
            return self
        else:
            return self.__class__(self._data, prefix=self._prefix + normalize_path(path), parent=self._parent)

    def copy(self, other):
        raise NotImplementedError()

    def get(self, rpath: Optional[_TPath] = None, *args, **kwargs) -> Any:
        return ht_get(self._data,  self._prefix + normalize_path(rpath),  *args, **kwargs)

    def put(self,  rpath:  Optional[_TPath],  *args, assign_if_exists=True, **kwargs):
        return ht_insert(self._data,  self._prefix + normalize_path(rpath),  *args, assign_if_exists=assign_if_exists or True, **kwargs)

    def insert(self, rpath: Optional[_TPath], v,  *args, **kwargs):
        return ht_insert(self._data,  self._prefix + normalize_path(rpath), v, *args, **kwargs)

    def update(self,  value, rpath: Optional[_TPath] = None, *args, **kwargs):
        raise NotImplementedError()

    def delete(self, rpath: Optional[_TPath] = None, *args, **kwargs):
        return ht_erase(self._data,  self._prefix + normalize_path(rpath),  *args, **kwargs)

    def count(self,  rpath: Optional[_TPath] = None,  *args, **kwargs) -> int:
        return ht_count(self._data,  self._prefix + normalize_path(rpath),  *args, **kwargs)

    def contains(self, v, rpath=None,  *args, **kwargs) -> bool:
        return ht_contains(self._data,  self._prefix + normalize_path(rpath),  *args, **kwargs)

    def call(self,   rpath: Optional[_TPath], *args, **kwargs) -> Any:
        obj = self.get(rpath, _not_found_)
        if callable(obj):
            res = obj(*args, **kwargs)
        elif len(args)+len(kwargs) == 0:
            res = obj
        else:
            raise TypeError(f"'{type(obj)}' is not callable")
        return res

    def equal(self, other) -> bool:
        return ht_compare(self.get(), other)

    def iter(self, *args, **kwargs):
        obj = self.get(*args, **kwargs)

        if isinstance(obj, Entry):
            yield from obj.iter()
        elif isinstance(obj, (collections.abc.Mapping, collections.abc.MutableSequence)):
            yield from obj
        else:
            raise NotImplementedError(type(obj))

    def items(self,  *args, **kwargs):
        obj = self.get(*args, **kwargs)
        if isinstance(obj, collections.abc.Mapping):
            yield from obj.items()
        elif isinstance(obj, collections.abc.MutableSequence):
            yield from enumerate(obj)
        elif isinstance(obj, Entry):
            yield from obj.items()
        else:
            raise TypeError(type(obj))

    def values(self,  *args, **kwargs):
        obj = self.get(*args, **kwargs)
        if isinstance(obj, collections.abc.Mapping):
            yield from obj.values()
        elif isinstance(obj, collections.abc.MutableSequence):
            yield from obj
        elif isinstance(obj, Entry):
            yield from []
        else:
            yield obj

    def keys(self,  *args, **kwargs):
        obj = self.get(*args, **kwargs)
        if isinstance(obj, collections.abc.Mapping):
            yield from obj.keys()
        elif isinstance(obj, collections.abc.MutableSequence):
            yield from range(len(obj))
        else:
            raise NotImplementedError()

    def __serialize__(self, *args, **kwargs):
        return [v for v in self.values(*args, **kwargs)]


class EntryWrapper(Entry):

    def __init__(self, d: Entry, *args, **kwargs):
        super().__init__(**kwargs)
        self._target = d

    def get(self, rpath: Optional[_TPath] = None, *args, default_value=_not_defined_, **kwargs):
        res = self._target.get(rpath, default_value=_not_found_)
        if res is _not_found_:
            res = super().get(rpath, default_value=default_value)
        return res

    def put(self, value,  path: Optional[_TPath] = None, **kwargs):
        super().put(value, path, **kwargs)

    def __len__(self) -> int:
        try:
            n = self._target.count()
        except Exception:
            n = super().count()
        return n

    def __contains__(self, key) -> bool:
        return self._target.contains(key) or super().contains(key)

    def __iter__(self):
        yield from self._target.iter()
        yield from super().iter()

    #  def get(self, path=[], *args, default_value=_not_found_, **kwargs):
    #     path = self._prefix + normalize_path(path)
    #     obj = self._data
    #     if obj is None:
    #         obj = self._parent
    #     for p in path:
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
    #             raise KeyError(path)
    #     return obj

    # def put(self,  path, value, *args, **kwargs):
    #     path = self._prefix + normalize_path(path)
    #     obj = self._data
    #     if len(path) == 0:
    #         return obj
    #     for p in path[:-1]:
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
    #     if hasattr(obj, path[-1]):
    #         setattr(obj, path[-1], value)
    #     else:
    #         obj[path[-1]] = value

    #     return obj[path[-1]]
