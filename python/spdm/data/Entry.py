
import collections
import collections.abc
import pprint
from typing import (Any, Generic, Iterator, Mapping, MutableMapping, Optional, Tuple,
                    MutableSequence, Sequence, Type, TypeVar, Union, get_args)

import numpy as np
from numpy.lib.arraysetops import isin
from spdm.util.utilities import normalize_path

from ..util.logger import logger

from ..util.utilities import _not_defined_, _not_found_, serialize

_next_ = object()
_last_ = object()

_TPath = TypeVar("_TPath", str, float, slice, Sequence)
_TKey = TypeVar('_TKey', int, str)
_TIndex = TypeVar('_TIndex', int, slice)


class Entry(object):

    _DICT_TYPE_ = dict
    _LIST_TYPE_ = list

    def __init__(self, data=None,  *args, prefix=None, parent=None, writable=True,   **kwargs):
        super().__init__()
        self._data = data
        self._parent = parent
        self._prefix = normalize_path(prefix)
        self._writable = writable

    @property
    def writable(self):
        return self._writable

    @property
    def data(self):
        return self._data

    @property
    def parent(self):
        return self._parent

    @property
    def prefix(self):
        return self._prefix

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
        # if isinstance(other, LazyProxy):
        #     other = other.__real_value__()
        # el
        if isinstance(other, Entry):
            other = other.entry.__real_value__()

        if isinstance(other, collections.abc.Mapping):
            for k, v in other.items():
                self._data[k] = v
        elif isinstance(other, collections.abc.MutableSequence):
            self._data.extend(other)
        else:
            raise ValueError(f"Can not copy {type(other)}!")

    def put(self,  value: Any, rpath:  Optional[_TPath] = None):
        path = self._prefix+normalize_path(rpath)

        if len(path) == 0 and self._data is None:
            self._data = value
            return self._data

        if self._data is None or self._data is _not_found_:
            self._data = Entry._DICT_TYPE_() if isinstance(path[0], str) else Entry._LIST_TYPE_()

        obj = self._data

        for idx, key in enumerate(path):
            obj = getattr(obj, "_entry", obj)

            if idx == len(path)-1:
                child = value
            else:
                child = Entry._DICT_TYPE_() if isinstance(path[idx+1], str) else Entry._LIST_TYPE_()

            if isinstance(obj, Entry):
                obj.put(path[idx:], child)
                # obj.put(key, child)

            elif isinstance(obj, collections.abc.MutableMapping):
                if not isinstance(key, str):
                    raise TypeError(f"mapping indices must be str, not {key}")
                elif idx == len(path)-1:
                    obj[key] = child
                    obj = child
                else:
                    tmp = obj.setdefault(key, child)
                    if tmp is None or tmp is _not_found_:
                        obj[key] = child
                        tmp = obj[key]
                    obj = tmp
            elif isinstance(obj, collections.abc.MutableSequence):
                if key is _next_:
                    obj.append(child)
                    obj = obj[-1]
                elif isinstance(key, (int, slice)):
                    tmp = obj[key]
                    if tmp is None:
                        obj[key] = child
                        obj = obj[key]
                    else:
                        obj = tmp
                else:
                    raise TypeError(f"list indices must be integers or slices, not {type(key).__name__}")
            else:
                raise TypeError(f"Can not insert data to {path[:idx]}! type={obj}")

        if rpath is None:
            self._data = self.get()
            self._prefix = []

    def get(self, rpath: Optional[_TPath] = None, default_value=_not_defined_) -> Any:
        path = self._prefix + normalize_path(rpath)

        obj = self._data
        suffix = None
        for idx, key in enumerate(path):
            if hasattr(obj, "_entry"):
                if obj._entry._data == self._data and obj._entry._prefix == path[:idx]:
                    suffix = path
                    obj = obj._entry._data
                    break
                else:
                    obj = obj._entry.get(path[idx:])
                break
            elif key is _next_:
                obj.append(_not_found_)
                obj = Entry(obj, prefix=[len(obj)-1] + path[idx:])
                break
            elif isinstance(obj, collections.abc.Mapping):
                if not isinstance(key, str):
                    raise TypeError(f"mapping indices must be str, not {type(key).__name__}! \"{path}\"")
                tmp = obj.get(key, _not_found_)
                if tmp is _not_found_:
                    suffix = path[idx:]
                    break
                obj = tmp
            elif isinstance(obj, collections.abc.MutableSequence):
                if not isinstance(key, (int, slice)):
                    raise TypeError(
                        f"list indices must be integers or slices, not {type(key).__name__}! \"{path[:idx+1]}\" {type(obj)}")
                elif isinstance(key, int) and isinstance(self._data, collections.abc.MutableSequence) and key > len(self._data):
                    raise IndexError(f"Out of range! {key} > {len(self._data)}")
                obj = obj[key]

        if rpath is None:
            self._data = obj
            self._prefix = []

        if suffix is None:
            return obj
        elif default_value is _not_defined_:
            return Entry(obj, prefix=suffix)
        else:
            return default_value

    def insert(self,   v, rpath: Optional[_TPath] = None, *args, **kwargs):
        path = self._prefix + normalize_path(rpath)
        try:
            parent = self.get(path[:-1])
        except KeyError:
            parent = None
        idx = path[-1]
        if parent is not None and idx in parent:
            pass
        elif parent is not None and idx not in parent:
            parent[idx] = v
        elif isinstance(idx, str):
            parent = self.put(path[:-1], {})
            parent[idx] = v
        elif type(path[-1]) is int and path[-1] <= 0:
            parent = self.put(path[:-1], [])
            idx = 0
            parent[idx] = v
        logger.debug(path)
        return parent[idx]

    def update(self,  v, path: Optional[_TPath] = None, *args, **kwargs):
        raise NotImplementedError()

    def delete(self, path: Optional[_TPath] = None, *args, **kwargs):
        path = self._prefix + normalize_path(path)

        if len(path) > 1:
            obj = self.get(path[:-1], *args, **kwargs)
        else:
            obj = self._data
        if hasattr(obj, path[-1]):
            delattr(obj, path[-1])
        else:
            del obj[path[-1]]

    def count(self,    *args, **kwargs) -> int:
        res = self.get(*args, **kwargs)
        if isinstance(res, Entry):
            return 0
        elif isinstance(res, (str, int, float, np.ndarray)):
            return 1
        elif isinstance(res, (collections.abc.Sequence, collections.abc.Mapping)):
            return len(res)
        else:
            raise TypeError(f"Not countable! {type(res)}")

    def contains(self, v,  *args, **kwargs) -> bool:
        return v in self.get(*args, **kwargs)

    def call(self,   rpath: Optional[_TPath], *args, **kwargs) -> Any:
        obj = self.get(rpath)
        if callable(obj):
            res = obj(*args, **kwargs)
        elif len(args)+len(kwargs) == 0:
            res = obj
        else:
            raise TypeError(f"'{type(obj)}' is not callable")

        return res

    def push_back(self,  v=None, rpath: Optional[_TPath] = None, *args, **kwargs):
        parent = self.insert([], rpath, *args, **kwargs)
        parent.append(v or {})
        return rpath+[len(parent)-1]

    def pop_back(self,  rpath: Optional[_TPath] = None, *args, **kwargs):
        obj = self.get(*args, **kwargs)
        res = None
        if obj is None:
            pass
        elif isinstance(obj, collections.abc.MutableSequence):
            res = obj[-1]
            obj.pop()
        else:
            raise KeyError(rpath)

        return res

    def equal(self, other) -> bool:
        obj = self.get(None)
        return (isinstance(obj, Entry) and other is None) or (obj == other)

    def iter(self, *args, default_value=None, **kwargs):
        obj = self.get(*args, default_value=default_value if default_value is not None else [], **kwargs)

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
