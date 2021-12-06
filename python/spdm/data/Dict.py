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
from .Entry import Entry, EntryChain
from .Node import Node, _TKey, _TObject

_T = TypeVar("_T")
_TDict = TypeVar('_TDict', bound='Dict')


class Dict(Container[_TObject], Mapping[str, _TObject]):

    def __init__(self, cache: Mapping = _undefined_,  /,  **kwargs):
        if cache is _undefined_:
            cache = kwargs
        elif len(kwargs) > 0:

            cache = EntryChain([kwargs, cache])

        super().__init__(cache)

    @property
    def _is_dict(self) -> bool:
        return True

    def __serialize__(self) -> Mapping:
        return {k: serialize(v) for k, v in self._entry.first_child()}

    def __getitem__(self, key: str) -> _TObject:
        return self._post_process(self._entry.child(key).pull(), key=key)

    def __setitem__(self, key: str, value: _T) -> None:
        return self._entry.child(key).push(self._pre_process(value))

    def __delitem__(self, key: str) -> None:
        self._entry.child(key).erase()

    def __iter__(self) -> Iterator[str]:
        yield from self._entry.first_child()

    def __eq__(self, o: Any) -> bool:
        return self._entry.equal(o)

    def __contains__(self, key) -> bool:
        return self._entry.child(key).exists

    def __iadd__(self, other) -> _TDict:
        return super().__iadd__(other)
        self

    def __ior__(self, other) -> _TDict:
        return super().__ior__(other)

    def __len__(self) -> int:
        return len(self._data)

    def update(self, d) -> _TDict:
        """Update the dictionary with the key/value pairs from other, overwriting existing keys. Return self.

        Args:
            d (Mapping): [description]

        Returns:
            _T: [description]
        """
        self._entry.push(d, update=True)
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

    def setdefault(self, key, *args) -> Any:
        """If key is in the dictionary, return its value. If not, insert key with a value of default and return default. default defaults to None.

        Args:
            path ([type]): [description]
            default_value ([type], optional): [description]. Defaults to _undefined_.

        Returns:
            Any: [description]
        """
        return self._post_process(self._entry.child(key).setdefault(*args), key=key)

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
