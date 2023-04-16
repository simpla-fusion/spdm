from __future__ import annotations

import collections
import collections.abc
import typing

from ..common.tags import _not_found_
from .Container import Container
from .Entry import Entry, EntryChain, as_entry
from .Node import Node
from .Path import Path

_T = typing.TypeVar("_T")
_TObject = typing.TypeVar("_TObject")


class Dict(Container[_TObject], typing.Mapping[str, _TObject]):
    def __init__(self, *args,   cache=None,  **kwargs):
        super().__init__(*args,   **kwargs)
        self._cache = {} if cache is None else cache

    def duplicate(self) -> Container:
        other: Dict[_TObject] = super().duplicate()  # type:ignore
        other._cache = self._cache
        return other

    @property
    def _is_dict(self) -> bool:
        return True

    def __iter__(self) -> typing.Generator[typing.Any, None, None]:
        yield from self.keys()

    def items(self) -> typing.Generator[typing.Tuple[typing.Any, typing.Any], None, None]:
        for key, value in self._entry.first_child():
            yield key, self._as_child(key, default_value=value)

    def keys(self) -> typing.Generator[typing.Any, None, None]:
        for key, _ in self._entry.first_child():
            yield key

    # def values(self) -> typing.Generator[typing.Any, None, None]:
    #     for key, value in self._entry.first_child():
    #         yield self._post_process(value, key=key)

    def __iadd__(self, other: typing.Mapping) -> Dict:
        return self.update(other)

    def __ior__(self, other) -> Dict:
        return self.update(other, force=False)

    def _as_child(self, key: str, value=_not_found_,
                  *args, **kwargs) -> _TObject:

        if value is _not_found_ and isinstance(key, str):
            value = self._cache.get(key, _not_found_)

        n_value = super()._as_child(key, value, *args, **kwargs)

        if isinstance(key, str):  # and n_value is not value:
            self._cache[key] = n_value

        return n_value

    def update(self, d, *args, **kwargs) -> Dict:
        """Update the dictionary with the key/value pairs from other, overwriting existing keys.
           Return self.

        Args:
            d (Mapping): [description]

        Returns:
            _T: [description]
        """
        self._entry.update(d, *args, **kwargs)
        return self

    # def get(self, key,  default_value=None) -> typing.Any:
    #     """Return the value for key if key is in the dictionary, else default.
    #        If default is not given, it defaults to None, so that this method never raises a KeyError.

    #     Args:
    #         path ([type]): [description]
    #         default ([type], optional): [description]. Defaults to None.

    #     Returns:
    #         Any: [description]
    #     """
    #     return self._post_process(self._entry.child(key), default_value=default_value)

    # def setdefault(self, key, value) -> typing.Any:
    #     """If key is in the dictionary, return its value.
    #        If not, insert key with a value of default and return default. default defaults to None.

    #     Args:
    #         path ([type]): [description]
    #         default_value ([type], optional): [description]. Defaults to None.

    #     Returns:
    #         Any: [description]
    #     """
    #     return self._post_process(self._entry.child(key).update({Path.tags.setdefault: value}), key=key)

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


Node._MAPPING_TYPE_ = Dict


def chain_map(*args, **kwargs) -> collections.ChainMap:
    d = []
    for a in args:
        if isinstance(d, collections.abc.Mapping):
            d.append(a)
        elif isinstance(d, Entry):
            d.append(Dict(a))
    return collections.ChainMap(*d, **kwargs)
