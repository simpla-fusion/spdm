
import pprint
import collections
import numpy as np
from spdm.util.sp_export import sp_find_module
from spdm.util.logger import logger


def load_ndarray(desc, value, *args, **kwargs):
    if isinstance(value, np.ndarray):
        return value
    else:
        return NotImplemented


class DataObject(object):
    @staticmethod
    def __new__(cls,  desc, value=None, *args, **kwargs):
        if cls is not DataObject:
            return super(cls, DataObject).__new__(desc, value, *args, **kwargs)

        if isinstance(desc, str):
            desc = {"schema": desc}

        if not isinstance(desc, collections.abc.Mapping):
            raise TypeError(f"Illegal type! 'desc' {type(desc)}")

        d_schema = desc.get("schema", "string")

        desc["schema"] = d_schema

        if value is None:
            value = desc.get("default", None)

        if d_schema == "integer":
            n_obj = int(value)
        elif d_schema == "float":
            n_obj = float(value)
        elif d_schema == "string":
            n_obj = str(value)
        elif d_schema == "ndarray":
            n_obj = load_ndarray(desc, value, *args, **kwargs)
        else:
            mod_path = f"{__package__}.data_object.{d_schema.replace('/','.')}"
            logger.debug(mod_path)
            n_cls = sp_find_module(mod_path)

            if hasattr(n_cls, "__new__"):
                n_obj = n_cls.__new__(n_cls, desc, value, *args, **kwargs)
            else:
                n_obj = object.__new__(n_cls)

        return n_obj

    def __init__(self, desc, value=None, *args, **kwargs):
        self._desc = desc

    def __repr__(self):
        return pprint.pformat(getattr(self, "_desc", self.__class__.__name__))
