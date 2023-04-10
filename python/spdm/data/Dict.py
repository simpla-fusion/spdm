import collections
import collections.abc
import typing

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from ..util.misc import serialize
from .Container import Container
from .Entry import Entry, as_entry
from .Node import Node

_T = typing.TypeVar("_T")
_TKey = typing.TypeVar("_TKey")
_TObject = typing.TypeVar("_TObject")
Dict = typing.TypeVar('Dict', bound='Dict')


class Dict(Container[str, _TObject]):

    def __init__(self, cache: typing.Optional[typing.Mapping] = None,  /,  **kwargs):
        if cache not in (None, _not_found_, None):
            super().__init__(cache)
        else:
            super().__init__({})

        self.update(kwargs)

    @property
    def _is_dict(self) -> bool:
        return True

    def __serialize__(self) -> dict:
        return {k: serialize(v) for k, v in self._entry.first_child()}

    def __setitem__(self, key, value: _T) -> _T:
        self._entry.child(key).insert(self._pre_process(value))
        return value

    def __getitem__(self, key) -> typing.Any:
        return self._post_process(self._entry.child(key), key=key)

    def __delitem__(self, key) -> bool:
        return self._entry.child(key).remove() > 0

    def __contains__(self, key) -> bool:
        return self._entry.child(key).exists

    def __eq__(self, other) -> bool:
        return self._entry.equal(other)

    def __len__(self) -> int:
        return self._entry.count

    def __iter__(self) -> typing.Generator[typing.Any, None, None]:
        yield from self.keys()

    def items(self) -> typing.Generator[typing.Tuple[typing.Any, typing.Any], None, None]:
        for key, value in self._entry.first_child():
            yield key, self._post_process(value, key=key)

    def keys(self) -> typing.Generator[typing.Any, None, None]:
        for key, _ in self._entry.first_child():
            yield key

    def values(self) -> typing.Generator[typing.Any, None, None]:
        for key, value in self._entry.first_child():
            yield self._post_process(value, key=key)

    def __iadd__(self, other: typing.Mapping) -> Dict:
        return self.update(other)

    def __ior__(self, other) -> Dict:
        return self.update(other, force=False)

    class DictAsEntry(Entry):
        def __init__(self, cache: Dict, **kwargs):
            super().__init__(cache, **kwargs)

        def pull(self, default=...) -> typing.Any:
            if len(self._path) > 0 and isinstance(self._path[0], str):
                obj = getattr(self._cache, self._path[0], _not_found_)
                if obj is _not_found_:
                    return self._cache._entry.child(self._path).query(default_value=default)
                elif len(self._path) == 1:
                    return obj
                else:
                    obj = as_entry(obj).child(self._path[1:]).query(default_value=_not_found_)
                    if obj in [_not_found_, None]:
                        return default
                    else:
                        self._cache = obj
                        self._path.clear()
                        return self._cache
            else:
                return self._cache.get(self._path, default)

        def push(self, value:  typing.Any, **kwargs) -> None:
            self._cache.push(value, **kwargs)

        def erase(self) -> bool:
            if len(self._path) == 1 and not isinstance(self._path[0], str) and self._path[0] in self._cache._properties:
                delattr(self._cache, self._path[0])
            return self._cache.erase(self._path)

    def __entry__(self) -> Entry:
        if self.__class__ is not Dict or getattr(self, "__orig_class__", _not_found_) is not _not_found_:
            return Dict.DictAsEntry(self)
        else:
            return self._entry

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

    def get(self, key,  default=None) -> typing.Any:
        """Return the value for key if key is in the dictionary, else default. 
           If default is not given, it defaults to None, so that this method never raises a KeyError.

        Args:
            path ([type]): [description]
            default ([type], optional): [description]. Defaults to None.

        Returns:
            Any: [description]
        """
        return self._post_process(self._entry.child(key).pull(default), key=key)

    def setdefault(self, key, value) -> typing.Any:
        """If key is in the dictionary, return its value. 
           If not, insert key with a value of default and return default. default defaults to None.

        Args:
            path ([type]): [description]
            default_value ([type], optional): [description]. Defaults to None.

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


Node._MAPPING_TYPE_ = Dict


def chain_map(*args, **kwargs) -> collections.ChainMap:
    d = []
    for a in args:
        if isinstance(d, collections.abc.Mapping):
            d.append(a)
        elif isinstance(d, Entry):
            d.append(Dict(a))
    return collections.ChainMap(*d, **kwargs)
