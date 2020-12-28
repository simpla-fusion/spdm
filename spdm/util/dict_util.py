import collections
from .AttributeTree import AttributeTree


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
        except Exception:
            res = None
        else:
            changed = True
        return res, changed

    if isinstance(obj, AttributeTree):
        data = obj.__data__
    else:
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

    if isinstance(obj, AttributeTree):
        obj.__data__ = data
    else:
        obj = data

    return obj, changed


def format_string_recursive(obj, mapping=None):

    if isinstance(mapping, AttributeTree):
        mapping = mapping.__as_native__()

    class DefaultDict(dict):
        def __missing__(self, key):
            return '{'+key+'}'

    d = DefaultDict(mapping)
    res, _ = tree_apply_recursive(obj, lambda s, _envs=d: s.format_map(d), str)
    return res
