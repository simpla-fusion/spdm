import functools

from .Node import Node


class AttributeTree(Node):

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args,   **kwargs)

    def __getattr__(self, k):
        if k.startswith("_"):
            return super().__getattr__(k)
        else:
            res = getattr(self.__class__, k, None)
            if res is None:
                res = self.__getitem__(k)
            elif isinstance(res, property):
                res = getattr(res, "fget")(self)
            elif isinstance(res, functools.cached_property):
                res = res.__get__(self)
            return res

    def __setattr__(self, k, v):
        if k.startswith("_"):
            super().__setattr__(k, v)
        else:
            res = getattr(self.__class__, k, None)
            if res is None:
                self.__setitem__(k, v)
            elif isinstance(res, property):
                res.fset(self, k, v)
            else:
                raise AttributeError(f"Can not set attribute {k}!")

    def __delattr__(self, k):
        if k.startswith("_"):
            super().__delattr__(k)
        else:
            res = getattr(self.__class__, k, None)
            if res is None:
                self.__delitem__(k)
            elif isinstance(res, property):
                res.fdel(self, k)
            else:
                raise AttributeError(f"Can not delete attribute {k}!")
