import functools


def _getattr(self, k):
    if k.startswith("_"):
        return self.__dict__.get(k, None)
    else:
        res = getattr(self.__class__, k, None)
        if res is None:
            res = self.__getitem__(k)
        elif isinstance(res, property):
            res = getattr(res, "fget")(self)
        elif isinstance(res, functools.cached_property):
            res = res.__get__(self)
        return res


def _setattr(self, k, v):
    if k.startswith("_"):
        self.__dict__[k] = v
    else:
        res = getattr(self.__class__, k, None)
        if res is None:
            self.__setitem__(k, v)
        elif isinstance(res, property):
            res.fset(self, k, v)
        else:
            raise AttributeError(f"Can not set attribute {k}!")


def _delattr(self, k):
    if k.startswith("_"):
        del self.__dict__[k]
    else:
        res = getattr(self.__class__, k, None)
        if res is None:
            self.__delitem__(k)
        elif isinstance(res, property):
            res.fdel(self, k)
        else:
            raise AttributeError(f"Can not delete attribute {k}!")


def as_attribute_tree(cls, *args, **kwargs):
    n_cls = type(f"{cls.__name__}__with_attr__", (cls,), {
        "__getattr__": _getattr,
        "__setattr__": _setattr,
        "__delattr__": _delattr,
    })

    return n_cls
