from __future__ import annotations

import collections
import collections.abc
import typing
from ..utils.logger import logger
from ..utils.tags import _not_found_
from .Container import Container
from .Entry import Entry, EntryChain, as_entry
from .HTree import Node
from .Path import Path

_T = typing.TypeVar("_T")


class Dict(Container[_T], typing.MutableMapping[str, _T]):
    """
        Dict
        ----
        Dict 是一个字典容器，它继承自 Container，因此它具有 Container 的所有特性。
        除此之外，它还具有 dict 的所有特性，包括：
        - keys
        - values
        - items
        - get
        - setdefault       
        - clear
        - update
        - copy
    """

    def __init__(self, d=None, *args, ** kwargs):
        super().__init__(d if d is not None else {}, *args,   **kwargs)

    def __type_hint__(self, key: str = None) -> typing.Type:
        type_hint = _not_found_
        if isinstance(key, str):
            t_hints = typing.get_type_hints(self.__class__)
            type_hint = t_hints.get(key, _not_found_)

        if type_hint is _not_found_:
            type_hint = super().__type_hint__()
        return type_hint

    def __iter__(self) -> typing.Generator[typing.Any, None, None]:
        yield from self.keys()

    def items(self) -> typing.Generator[typing.Tuple[typing.Any, typing.Any], None, None]:
        for key, value in self._entry.children:
            yield key, self.as_child(key, value)

    def keys(self) -> typing.Generator[typing.Any, None, None]:
        for key, _ in self._entry.children:
            yield key

    def values(self) -> typing.Generator[typing.Any, None, None]:
        raise NotImplementedError()

    def __iadd__(self, other: typing.Mapping) -> Dict:
        return self._update(other)

    def __ior__(self, other) -> Dict:
        return self.update(other, force=False)

    # def _as_child(self, key: str,  value=_not_found_, *args, default_value=_not_found_, **kwargs) -> _T:

    #     if self._default_value is not _not_found_ and isinstance(key, str):
    #         # _as_child 中的 default_value 来自 sp_property 的 type_hint， self._default_value 来自 entry,
    #         # 所以优先采用 self._default_value
    #         default_value = self._default_value.get(key, _not_found_)

    #     if (value is _not_found_ or value is None) and key is not None:
    #         value = self._cache.get(key, _not_found_)

    #     n_value = super()._as_child(key, value, *args, default_value=default_value, **kwargs)

    #     if n_value is not self._cache.get(key, None):
    #         self._cache[key] = n_value

    #     return n_value

    def update(self,   *args, **kwargs) -> Dict:
        """Update the dictionary with the key/value pairs from other, overwriting existing keys.
           Return self.

        Args:
            d (Mapping): [description]

        Returns:
            _T: [description]
        """
        self._entry.update(*args, **kwargs)
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
    #     return self._post_process(self.__entry__().child(key), default_value=default_value)

    # def setdefault(self, key, value) -> typing.Any:
    #     """If key is in the dictionary, return its value.
    #        If not, insert key with a value of default and return default. default defaults to None.

    #     Args:
    #         path ([type]): [description]
    #         default_value ([type], optional): [description]. Defaults to None.

    #     Returns:
    #         Any: [description]
    #     """
    #     return self._post_process(self.__entry__().child(key).update({Path.tags.setdefault: value}), key=key)

    # def _as_dict(self) -> Mapping:
    #     cls = self.__class__
    #     if cls is Dict:
    #         return self.__entry__()._data
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
    #                 v = self.__entry__().find(k)
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
    #         self.__entry__() = Entry(data, parent=self.__entry__().parent)
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
