from .LazyProxy import LazyProxy
import collections
import pprint
from .logger import logger


class AttributeTree(dict):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def entry(self):
        return LazyProxy(self)

    class __lazy_proxy__:
        @staticmethod
        def put(obj, path, value, *args, **kwargs):
            if len(path) == 0:
                raise KeyError("Empty path")
            for idx, p in enumerate(path[:-1]):
                if isinstance(obj, AttributeTree):
                    obj = obj.setdefault(p, AttributeTree())
                elif not isinstance(p, str) and isinstance(obj, list):
                    obj = obj[p]
                else:
                    raise KeyError(".".join(path[:idx+1]))
            if hasattr(obj, path[-1]):
                setattr(obj, path[-1], value)
            else:
                obj[path[-1]] = value

            # if len(path) > 0:
            #     raise path

            return None

        @staticmethod
        def get(obj, path,  *args, **kwargs):
            if len(path) == 0:
                return obj

            for idx, p in enumerate(path):
                if isinstance(obj, AttributeTree):
                    obj = obj.get(p, None)
                elif isinstance(obj, list):
                    try:
                        obj = obj[p]
                    except IndexError:
                        obj = None
                else:
                    obj = getattr(obj, p, None)

                if obj is None:
                    raise KeyError(f"Illegal path '{'.'.join(path[:idx+1])}'! ")

            return obj

        @staticmethod
        def get_value(data, path, *args, **kwargs):
            return AttributeTree.__lazy_proxy__.get(data, path, *args, **kwargs)

        @staticmethod
        def delete(data,  path, *args, **kwargs):
            if len(path) > 1:
                obj = AttributeTree.__lazy_proxy__.get(data, path[:-1], *args, **kwargs)
            else:
                obj = data
            if hasattr(obj, path[-1]):
                delattr(obj, path[-1])
            else:
                del obj[path[-1]]

        @staticmethod
        def push_back(data,  path, value, *args, **kwargs):
            # if len(path) == 0:
            #     data.push_back(value)
            # else:
            # data.get(path).setdefault(path[-1], []).append(value)
            if len(path) > 0:
                obj = AttributeTree.__lazy_proxy__.get(data, path[:-1]).setdefault(path[-1], [])
            else:
                obj = data

            obj.append(value or AttributeTree())
            return path+[len(obj)-1]

        @staticmethod
        def count(data,  path, *args, **kwargs):
            obj = AttributeTree.__lazy_proxy__.get(data, path, *args, **kwargs)
            return len(obj)

        @staticmethod
        def contains(data,  path, v, *args, **kwargs):
            obj = AttributeTree.__lazy_proxy__.get(data, path, *args, **kwargs)
            return v in obj

        @staticmethod
        def iter(data,  path, *args, **kwargs):
            for obj in AttributeTree.__lazy_proxy__.get(data, path, *args, **kwargs):
                if type(obj) in (int, float, str):
                    yield obj
                else:
                    yield LazyProxy(obj)

        @staticmethod
        def call(data, path, *args, **kwargs):
            obj = AttributeTree.__lazy_proxy__.get(data, path)
            if callable(obj):
                return obj(*args, **kwargs)
            elif len(args)+len(kwargs) == 0:
                return obj
            else:
                raise TypeError(f"{obj.__class__.__name__} is not callable")


def attribute_tree(*args, **kwargs):
    return AttributeTree(*args, **kwargs).entry
