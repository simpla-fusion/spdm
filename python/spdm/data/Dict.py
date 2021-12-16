import collections
import collections.abc
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from ..util.dict_util import deep_merge_dict
from ..util.utilities import serialize

from .Container import Container
from .Entry import Entry, EntryChain, as_entry
from .Node import Node, _TKey, _TObject

_T = TypeVar("_T")
_TDict = TypeVar('_TDict', bound='Dict')


class Dict(Container[_TObject], Mapping[str, _TObject]):

    def __init__(self, cache: Mapping = _undefined_,  /,  **kwargs):
        if cache is _undefined_ or cache is _not_found_:
            cache = {}

        super().__init__(cache)

        if len(kwargs) > 0:
            self.update(kwargs)

    @property
    def _is_dict(self) -> bool:
        return True

    def __serialize__(self) -> Mapping:
        return {k: serialize(v) for k, v in self._entry.first_child()}

    def __getitem__(self, key) -> _TObject:
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: _T) -> None:
        return super().__setitem__(key, value)

    def __delitem__(self,  key) -> None:
        return super().__delitem__(key)

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[str]:
        yield from super().__iter__()
        # yield from self._entry.first_child()

    def __eq__(self, o: Any) -> bool:
        return self._entry.equal(o)

    def __contains__(self, key) -> bool:
        return self._entry.child(key).exists

    def __iadd__(self, other) -> _TDict:
        return super().__iadd__(other)
        self

    def __ior__(self, other) -> _TDict:
        return super().__ior__(other)

    class DictAsEntry(Entry):
        def __init__(self, cache, **kwargs):
            super().__init__(cache, **kwargs)

        def pull(self, default=...) -> Any:
            if len(self._path) > 0 and isinstance(self._path[0], str):
                obj = getattr(self._cache, self._path[0], _not_found_)
                if obj is _not_found_:
                    return self._cache._entry.child(self._path).pull(default)
                elif len(self._path) == 1:
                    return obj
                else:
                    obj = as_entry(obj).child(self._path[1:]).pull(_not_found_)
                    if obj in [_not_found_, _undefined_]:
                        return default
                    else:
                        self._cache = obj
                        self._path.reset()
                        return self._cache
            else:
                return self._cache._entry.child(self._path).pull(default)

        def push(self, value: Any, **kwargs) -> None:
            self._cache._entry.child(self._path).push(value, **kwargs)

        def erase(self) -> bool:
            if len(self._path) == 1 and not isinstance(self._path[0], str) and self._path[0] in self._cache._properties:
                delattr(self._cache, self._path[0])
            return self._cache._entry.child(self._path).erase()

    def __entry__(self) -> Entry:
        if self.__class__ is not Dict or getattr(self, "__orig_class__", _not_found_) is not _not_found_:
            return Dict.DictAsEntry(self)
        else:
            return self._entry

    def update(self, d) -> _TDict:
        """Update the dictionary with the key/value pairs from other, overwriting existing keys. Return self.

        Args:
            d (Mapping): [description]

        Returns:
            _T: [description]
        """
        self._entry.update(d)
        return self

    def get(self, key,  default=_undefined_) -> Any:
        """Return the value for key if key is in the dictionary, else default. If default is not given, it defaults to None, so that this method never raises a KeyError.

        Args:
            path ([type]): [description]
            default ([type], optional): [description]. Defaults to _undefined_.

        Returns:
            Any: [description]
        """
        return self._post_process(self._entry.child(key).pull(default), key=key)

    def setdefault(self, key, value) -> Any:
        """If key is in the dictionary, return its value. If not, insert key with a value of default and return default. default defaults to None.

        Args:
            path ([type]): [description]
            default_value ([type], optional): [description]. Defaults to _undefined_.

        Returns:
            Any: [description]
        """
        return self._post_process(self._entry.child(key).push(value), key=key)

    # def _as_dict(self) -> Mapping:
    #     cls = self.__class__
    #     if cls is Dict:
    #         return self._entry._data
    #     else:
    #         properties = set([k for k in self.__dir__() if not k.startswith('_')])
    #         res = {}
    #         for k in properties:
    #             prop = getattr(cls, k, None)
    #             if inspect.isfunction(prop) or inspect.isclass(prop) or inspect.ismethod(prop):
    #                 continue
    #             elif isinstance(prop, cached_property):
    #                 v = prop.__get__(self)
    #             elif isinstance(prop, property):
    #                 v = prop.fget(self)
    #             else:
    #                 v = getattr(self, k, _not_found_)
    #             if v is _not_found_:
    #                 v = self._entry.find(k)
    #             if v is _not_found_ or isinstance(v, Entry):
    #                 continue
    #             # elif hasattr(v, "_serialize"):
    #             #     res[k] = v._serialize()
    #             # else:
    #             #     res[k] = serialize(v)
    #             res[k] = v
    #         return res
    # self.__reset__(d.keys())
    # def __reset__(self, d=None) -> None:
    #     if isinstance(d, str):
    #         return self.__reset__([d])
    #     elif d is None:
    #         return self.__reset__([d for k in dir(self) if not k.startswith("_")])
    #     elif isinstance(d, Mapping):
    #         properties = getattr(self.__class__, '_properties_', _not_found_)
    #         if properties is not _not_found_:
    #             data = {k: v for k, v in d.items() if k in properties}
    #         self._entry = Entry(data, parent=self._entry.parent)
    #         self.__reset__(d.keys())
    #     elif isinstance(d, Sequence):
    #         for key in d:
    #             if isinstance(key, str) and hasattr(self, key) and isinstance(getattr(self.__class__, key, _not_found_), functools.cached_property):
    #                 delattr(self, key)


def chain_map(*args, **kwargs) -> collections.ChainMap:
    d = []
    for a in args:
        if isinstance(d, collections.abc.Mapping):
            d.append(a)
        elif isinstance(d, Entry):
            d.append(Dict(a))
    return collections.ChainMap(*d, **kwargs)
