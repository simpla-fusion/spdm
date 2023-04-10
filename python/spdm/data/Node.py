from __future__ import annotations

import collections
import collections.abc
import dataclasses
import inspect
import typing

import numpy as np

from ..common.tags import _not_found_, _undefined_, tags
from .Entry import Entry, as_entry


class Node(object):
    """

    """

    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)
    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list

    def __new__(cls, d, *args, **kwargs):

        if hasattr(d, "__entry__"):
            if d.__entry__.is_sequence:
                n_cls = Node._SEQUENCE_TYPE_
            elif d.__entry__.is_mapping:
                n_cls = Node._MAPPING_TYPE_
            else:
                n_cls = cls
        elif isinstance(d, collections.abc.Sequence) and not isinstance(d, str):
            n_cls = Node._SEQUENCE_TYPE_
        elif isinstance(d, collections.abc.Mapping):
            n_cls = Node._MAPPING_TYPE_
        else:
            n_cls = cls

        if n_cls in (dict, list):
            return n_cls.__new__(n_cls)
        else:
            return object.__new__(n_cls)

    def __init__(self, data=None, *args, **kwargs) -> None:
        super().__init__()
        self._entry = as_entry(data)
        self._nid = ""
        self._parent = None

    @property
    def annotation(self) -> dict:
        return {"id": self.nid,   "type":  self._entry.__class__.__name__}

    @property
    def nid(self) -> str:
        return self._nid

    @property
    def __entry__(self) -> Entry:
        return self._entry

    def reset(self):
        self._entry.reset()

    def dump(self):
        return self.__serialize__()

    def __serialize__(self):
        return self._entry.dump()

    @property
    def value(self) -> typing.Any:
        return self.update_child(_undefined_, self._entry.query(default_value=_not_found_))

    def _pre_process(self, value: typing.Any, *args, **kwargs) -> typing.Any:
        return value

    def _post_process(self, value: typing.Any, key, *args,  ** kwargs) -> typing.Union[typing.Any, Node]:
        return self.update_child(key, value, *args,  ** kwargs)

    def validate(self, value, type_hint) -> bool:
        if value is _undefined_ or type_hint is _undefined_:
            return False
        else:
            v_orig_class = getattr(value, "__orig_class__", value.__class__)

            if inspect.isclass(type_hint) and inspect.isclass(v_orig_class) and issubclass(v_orig_class, type_hint):
                res = True
            elif typing.get_origin(type_hint) is not None and typing.get_origin(v_orig_class) is typing.get_origin(type_hint) and typing.get_args(v_orig_class) == typing.get_args(type_hint):
                res = True
            else:
                res = False
        return res

    def update_child(self,
                     key: typing.Any,
                     value: typing.Optional[typing.Any] = None,
                     type_hint=None,
                     default_value: typing.Optional[typing.Any] = None,
                     getter:  typing.Optional[typing.Callable] = None,
                     in_place=True,
                     force=True,
                     *args, **kwargs) -> typing.Union[typing.Any, Node]:

        is_changed = True

        if value is None and key is not None:
            value = self._entry.child(key).query(default_value=_not_found_)
            is_changed = value is _not_found_

        is_valid = self.validate(value, type_hint) if value is not _not_found_ else False

        if not is_valid:
            if getter is not None:
                value = getter(self)
            elif value is _undefined_:
                value = default_value
            is_changed = True
            is_valid = self.validate(value, type_hint)

        if is_valid:
            obj = value
        elif type_hint is _undefined_:
            if isinstance(value, (collections.abc.Sequence, collections.abc.Mapping, Entry)) and not isinstance(value, str):
                obj = Node(value, *args, **kwargs)
            else:
                obj = value
            # obj = value if not isinstance(value, Entry) else value.dump()
        elif type_hint in Node._PRIMARY_TYPE_:  # (int, float, bool, str):
            if isinstance(value, Entry):
                value = value.query(default_value=_not_found_)
            elif hasattr(value, "__entry__"):
                value = value.__entry__.__value__
            if value is _undefined_ or isinstance(value, Entry):
                raise TypeError(value)
            elif type_hint is np.ndarray:
                obj = np.asarray(value)
            elif isinstance(value, tags):
                raise ValueError(f"Tags is not a value! key={key} tags={value}")
            else:
                try:
                    obj = type_hint(value)
                except TypeError as err:
                    raise TypeError(f"Can't convert value {value} to {type_hint}") from err

        elif dataclasses.is_dataclass(type_hint):
            if isinstance(value, collections.abc.Mapping):
                obj = type_hint(**{k: value.get(k, None) for k in type_hint.__dataclass_fields__})
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                obj = type_hint(*value)
            else:
                obj = type_hint(value)
        elif inspect.isfunction(type_hint):
            obj = type_hint(value, *args,  **kwargs)
        elif inspect.isclass(type_hint):
            obj = type_hint.__new__(type_hint, value, *args, **kwargs)
            obj._parent = self
            obj.__init__(value, *args, **kwargs)
        elif typing.get_origin(type_hint) is not None:
            obj = type_hint(value, *args, **kwargs)
        else:
            raise NotImplementedError(type_hint)

        # elif hasattr(type_hint, '__origin__'):
            # if issubclass(type_hint.__origin__, Node):
            #     obj = type_hint(value, parent=parent, **kwargs)
            # else:
            #     obj = type_hint(value, **kwargs)
        # if inspect.isclass(type_hint):
        #     if issubclass(type_hint, Node):
        #         obj = type_hint(value, *args, parent=parent, **kwargs)
        # elif callable(type_hint):
        #     obj = type_hint(value, **kwargs)
        # else:
        #     if always_node:
        #         obj = Node(value, *args, parent=parent, **kwargs)
        #     logger.warning(f"Ignore type_hint={type(type_hint)}!")

        is_changed |= obj is not value

        ###################################################################

        if key is not _undefined_ and is_changed:
            if isinstance(obj, Entry) or isinstance(value, Entry):  # and self._entry._cache is value._cache:
                pass
            elif in_place and isinstance(key, (int, str)):
                self._entry.child(key).insert(obj)

        if isinstance(obj, Node):
            obj._parent = self

        return obj

        # if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        #     res = Node._SEQUENCE_TYPE_(value,  parent=parent, **kwargs)
        # elif isinstance(value, collections.abc.Mapping):
        #     res = Node._MAPPING_TYPE_(value,   parent=parent, **kwargs)
        # elif isinstance(value, Entry):
        #     if Node._LINK_TYPE_ is not None:
        #         res = Node._LINK_TYPE_(value,  parent=parent, **kwargs)
        #     else:
        #         res = Node(value,  parent=parent, **kwargs)
        # if isinstance(value, Node._PRIMARY_TYPE_) or isinstance(value, Node) or value in (None, _not_found_, _undefined_):
        #     return value
