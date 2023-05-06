import collections
import collections.abc
import dataclasses
import functools
import inspect
import operator
from copy import deepcopy
from typing import (Any, Callable, Generic, Iterator, Mapping, Sequence, Tuple,
                    Type, TypeVar, Union)

from ..utils.tags import _not_found_, _undefined_
from .Path import Path


def normal_get(obj, key, default=_not_found_):
    if key is None:
        return obj
    elif isinstance(key, (int, slice)) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        return obj[key]
    elif isinstance(key, (int, slice)) and isinstance(obj, collections.abc.Mapping):
        return {k: normal_get(v, key, default) for k, v in obj.items()}
    elif isinstance(key, str) and isinstance(obj, collections.abc.Mapping):
        return obj.get(key, default)
    elif isinstance(key, str) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        return [normal_get(v, key, default) for v in obj]
    elif isinstance(key, set):
        return {k: normal_get(obj, k, default) for k in key}
    elif isinstance(key, collections.abc.Sequence):
        return [normal_get(obj, k, default) for k in key]
    elif isinstance(key, collections.abc.Mapping):
        return {k: normal_get(obj, v, default) for k, v in key.items()}
    elif hasattr(obj, "get") and isinstance(key, str):
        return obj.get(key, default)
    else:
        raise NotImplementedError(key)


def normal_put(obj, key, value, update=False, extend=False):
    if hasattr(obj, "put"):
        obj.put(key, value, update=update, extend=extend)
    elif hasattr(obj, '_entry'):
        obj[key] = value
    elif (update or extend) and key is not _undefined_:
        tmp = normal_get(obj, key, _not_found_)
        if tmp is not _not_found_:
            normal_put(tmp, _undefined_, value, update=update, extend=extend)
        else:
            normal_put(obj, key, value)
    elif key is _undefined_:
        if isinstance(obj, collections.abc.Mapping) and isinstance(value, collections.abc.Mapping):
            for k, v in value.items():
                normal_put(obj, k, v, update=update)
        elif isinstance(obj, collections.abc.Sequence) and isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            obj.extend(value)
            # for k, v in enumerate(value):
            #     normal_set(obj, k, v, update=update)
        else:
            raise KeyError(type(value))
    elif isinstance(key, (int, str, slice)):
        obj[key] = value
    elif isinstance(key, collections.abc.Sequence):
        for i in key:
            normal_put(obj, i, value, update=update)
    elif isinstance(key, collections.abc.Mapping):
        for i, v in key.items():
            normal_put(obj, i, normal_get(value, v), update=update)
    else:
        raise NotImplementedError((obj, key))


def normal_filter(obj: Sequence, query,  only_first=True) -> Iterator[Tuple[int, Any]]:
    for idx, val in enumerate(obj):
        if normal_check(val, query):
            yield idx, val
            if only_first:
                break


def normal_check(obj, query, expect=None) -> bool:
    if query in [_undefined_, None, _not_found_]:
        return obj
    elif isinstance(query, str):
        if query[0] == '$':
            raise NotImplementedError(query)
            # return _op_tag(query, obj, expect)
        elif isinstance(obj, collections.abc.Mapping):
            return normal_get(obj, query, _not_found_) == expect
        elif hasattr(obj, "_entry"):
            return normal_get(obj._entry, query, _not_found_) == expect
        else:
            raise TypeError(query)

    elif isinstance(query, collections.abc.Mapping):
        return all([normal_check(obj, k, v) for k, v in query.items()])
    elif isinstance(query, collections.abc.Sequence):
        return all([normal_check(obj, k) for k in query])
    else:
        raise NotImplementedError(query)
