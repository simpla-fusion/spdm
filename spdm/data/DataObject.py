
import pprint
import collections


class DataObject(object):
    @staticmethod
    def __new__(cls,  desc, value=None, *args, **kwargs):
        if isinstance(desc, str):
            desc = {"type": desc}

        if not isinstance(desc, collections.abc.Mapping):
            raise TypeError(f"Illegal type! 'desc' {type(desc)}")

        if isinstance(value, (int, str)):
            n_obj = value
        elif value is None:
            n_obj = desc.get("default", None)
        else:
            n_obj = object.__new__(cls)

        return n_obj

    def __init__(self, desc, value=None, *args, **kwargs):
        self._desc = desc

    def __repr__(self):
        return pprint.pformat(self._desc)
