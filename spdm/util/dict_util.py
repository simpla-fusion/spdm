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

    if types is not None and isinstance(obj, types):
        try:
            res = op(obj)
        except Exception:
            res = None
            flag = False
        else:
            flag = True
        return res, flag
    elif isinstance(obj, AttributeTree):
        obj = obj.__data__

    if not isinstance(obj, str) and isinstance(obj, collections.abc.Sequence):
        for idx, v in enumerate(obj):
            new_v, changed = tree_apply_recursive(v, op, types)
            if changed:
                obj[idx] = new_v
    elif isinstance(obj, collections.abc.Mapping):
        for idx, v in obj.items():
            new_v, changed = tree_apply_recursive(v, op, types)
            if changed:
                obj[idx] = new_v

    return obj, False
