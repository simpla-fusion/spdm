import collections.abc
from copy import deepcopy
import typing
import numpy as np

from .logger import logger


class DefaultDict(dict):
    def __init__(self, default_factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._factory = default_factory

    def __missing__(self, k):
        v = self._factory(k)
        self.__setitem__(k,  v)
        return v


def tree_apply_recursive(obj, op, types=None):
    changed = False
    is_attr_tree = False
    if types is not None and isinstance(obj, types):
        try:
            res = op(obj)
        except Exception as error:
            res = None
        else:
            changed = True
        return res, changed

    data = obj

    changed = False

    if not isinstance(data, str) and isinstance(data, collections.abc.Sequence):
        for idx, v in enumerate(data):
            new_v, flag = tree_apply_recursive(v, op, types)
            if flag:
                data[idx] = new_v
                changed = True
    elif isinstance(data, collections.abc.Mapping):
        for idx, v in data.items():
            new_v, flag = tree_apply_recursive(v, op, types)
            if flag:
                data[idx] = new_v
                changed = True

    obj = data

    return obj, changed


def format_string_recursive(obj, mapping=None):
    class DefaultDict(dict):
        def __missing__(self, key):
            return '{'+key+'}'

    d = DefaultDict(mapping)
    res, _ = tree_apply_recursive(obj, lambda s, _envs=d: s.format_map(d), str)
    return res


def normalize_data(data, types=(int, float, str)):
    if isinstance(data, types):
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {k:  normalize_data(v, types) for k, v in data.items()}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        return [normalize_data(v) for v in data]
    else:
        return str(data)


def deep_merge_dict(first: dict | list, second: dict, level=-1, in_place=False) -> dict | list:
    if not in_place:
        first = deepcopy(first)

    if level == 0:
        return first
    elif isinstance(first, collections.abc.Sequence):
        if isinstance(second, collections.abc.Sequence):
            first.extent(second)
        else:
            first.append(second)
    elif isinstance(first, collections.abc.Mapping) and isinstance(second, collections.abc.Mapping):
        for k, v in second.items():
            d = first.get(k, None)
            if isinstance(d, collections.abc.Mapping):
                deep_merge_dict(d, v, level-1)
            elif d is None:  # or not isinstance(v, collections.abc.Mapping):
                first[k] = v
    elif second is None:
        pass
    else:
        raise TypeError(f"Can not merge dict with {type(second)}!")

    return first


def reduce_dict(d, **kwargs) -> typing.Dict:
    res = {}
    for v in d:
        deep_merge_dict(res, v, in_place=True, **kwargs)
    return res


def _recursive_get(obj, k):
    return obj if len(k) == 0 else _recursive_get(obj[k[0]], k[1:])


class DictTemplate:
    def __init__(self, tmpl, *args, **kwargs):
        self._template = tmpl

    def __missing__(self, key):
        return '{'+key+'}'

    def __getitem__(self, key):
        try:
            res = self._template[key]
        except (KeyError, IndexError):
            raise KeyError(key)

        return res

    def get(self, key, default_value=None):
        try:
            if isinstance(key, str):
                res = _recursive_get(self._template, key.split('.'))
            else:
                res = self._template[key]
        except (KeyError, IndexError):
            res = default_value
        return res

    def apply(self, data):
        return tree_apply_recursive(data, lambda s, _template=self: s.format_map(_template), str)[0]


def convert_to_named_tuple(d=None, ntuple=None, **kwargs):
    if d is None and len(kwargs) > 0:
        d = kwargs
    if d is None:
        return d
    elif hasattr(ntuple, "_fields") and isinstance(ntuple, type):
        return ntuple(*[try_get(d, k) for k in ntuple._fields])
    elif isinstance(d, collections.abc.Mapping):
        keys = [k.replace('$', 's_') for k in d.keys()]
        values = [convert_to_named_tuple(v) for v in d.values()]
        if not isinstance(ntuple, str):
            ntuple = "__"+("_".join(keys))
        ntuple = ntuple.replace('$', '_')
        return collections.namedtuple(ntuple, keys)(*values)
    elif isinstance(d, collections.abc.MutableSequence):
        return [convert_to_named_tuple(v) for v in d]
    else:
        return d


def as_native(d, enable_ndarray=True) -> typing.Union[str, bool, float, int, np.ndarray, dict, list]:
    """
        convert d to native data type str,bool,float, int, dict, list
    """
    if isinstance(d, (bool, int, float, str)):
        return d
    elif isinstance(d, np.ndarray):
        return d.tolist() if not enable_ndarray else d
    elif isinstance(d, collections.abc.Mapping):
        return {as_native(k): as_native(v, enable_ndarray=enable_ndarray) for k, v in d.items()}
    elif isinstance(d, collections.abc.Sequence):
        return [as_native(v, enable_ndarray=enable_ndarray) for v in d]

    else:
        logger.debug(type(d))
        return str(d)
