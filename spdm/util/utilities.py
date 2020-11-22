import collections
import functools
import importlib
import inspect
import io
import json
import os
import pathlib
import pkgutil
import pwd
import re
import sys
import re
from .logger import logger
from .LazyProxy import LazyProxy


class _Empty:
    pass


_empty = _Empty()


def whoami(obj=None):
    cls = getattr(obj, "__class__", None)
    if cls is not None:
        return f"{cls.__name__}.{inspect.stack()[1][3]}"
    else:
        return inspect.stack()[1][3]


def getitem(obj, key=None, default_value=None):
    if key is None:
        return obj
    elif hasattr(obj, "__get__"):
        return obj.__get__(key, default_value)
    elif hasattr(obj, "__getitem__"):
        return obj.__getitem__(key) or default_value
    else:
        return default_value


def setitem(obj, key, value):
    if hasattr(obj, "__setitem__"):
        return obj.__setitem__(key, value)
    else:
        raise KeyError(f"Can not setitem {key}")


def iteritems(obj):
    if obj is None:
        return []
    elif isinstance(obj, collections.abc.Sequence):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        return obj.items()
    else:
        raise TypeError(f"Can not apply 'iteritems' on {type(obj)}!")


def get_cls_file_path(cls):
    return pathlib.Path(inspect.getfile(cls)).parent


def load_pkg_data(pkgname, path):
    data = pkgutil.get_data(pkgname, path)
    return json.loads(data.decode("utf-8"))


def try_getattr_r(obj, path: str):
    if path is None or path == '':
        return obj, ''
    start = 0
    path = path.strip(".")
    s_len = len(path)
    while start >= 0 and start < s_len:
        pos = path.find('.', start)
        if pos < 0:
            pos = s_len
        if not hasattr(obj, path[start:pos]):
            break
        obj = getattr(obj, path[start: pos])
        start = pos+1
    return obj, path[start:]


def getattr_r(obj, path: str):
    # o, p = try_getattr_r(obj, path)

    # if p != '':
    #     raise KeyError(f"Can for find path {path}")
    if type(path) is str:
        path = path.split('.')

    o = obj
    for p in path:
        o = getattr(o, p, None)
        if o is None:
            break
            # raise KeyError(f"Can not get attribute {p} from {o}")
    return o


# def try_getitem_r(obj, path):
#     if path is None or path == '':
#         return obj, ''
#     start = 0
#     path = path.strip(".")
#     s_len = len(path)
#     while start >= 0 and start < s_len:
#         pos = path.find('.', start)
#         if pos < 0:
#             pos = s_len
#         if isinstance(obj, collections.abc.Mapping) and path[start:pos] in obj:
#             obj = obj.get(path[start: pos])
#             start = pos+1
#         else:
#             break
#     return obj, path[start:]


# def getitem_r(obj, path: str):
#     o, p = try_getitem_r(obj, path)
#     if p != '':
#         raise KeyError(f"Can for find path {path}")
#     return o


def as_file_fun(func=None,  *f_args, **f_kwargs):
    """ Function wrapper: Convert  first argument (as file path) to File object
    TODO salmon (20190915): specify the position/key of the file path argument
    """
    def _decorate(wrapped):
        @functools.wraps(wrapped)
        def _wrapper(fp, *args, **kwargs):
            if isinstance(fp, str):
                with open(fp, *f_args, **f_kwargs) as fid:
                    res = wrapped(fid, *args, **kwargs)
            elif isinstance(fp, pathlib.Path):
                with fp.open(*f_args, **f_kwargs) as fid:
                    res = wrapped(fid, *args, **kwargs)
            elif isinstance(fp, io.IOBase):
                res = func(fp, *args, **kwargs)
            else:
                raise TypeError(
                    f"Can not convert type({type(fp)}) to file-like object!")

            return res
        return _wrapper

    if func is None:
        return _decorate
    else:
        return _decorate(func)


def as_lazy_object(func,  *f_args, **f_kwargs):
    def _decorate(wrapped):
        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            def fun():
                return wrapped(*args, **kwargs)
            return LazyProxy(fun)
        return _wrapper

    if func is None:
        return _decorate
    else:
        return _decorate(func)


def getlogin():
    try:
        return os.getlogin()
    except Exception:
        return pwd.getpwuid(os.getuid())[0]


def _gusses_name(self, name_hint):
    count = sum(1 for k, v in self._graph.nodes.items()
                if k == name_hint or k.startswith(f"{name_hint}_"))
    if count > 0:
        name_hint = f"{name_hint}_{count}"
    return name_hint


def _try_insert(self, name_hint, node):
    return self.add_node(node, label=self._gusses_name(name_hint or node.__class__.__name__.lower()))


def deep_update_dict(first, second, level=-1):
    if level == 0:
        return
    elif isinstance(first, collections.abc.Sequence):
        if isinstance(second, collections.abc.Sequence):
            first.extent(second)
        else:
            first.append(second)
    elif isinstance(first, collections.abc.Mapping) and isinstance(second, collections.abc.Mapping):
        for k, v in second.items():
            if isinstance(first.get(k, None), collections.abc.Mapping):
                deep_update_dict(first[k], v, level-1)
            else:
                first[k] = v
    elif second is None:
        pass
    else:
        raise TypeError(f"Can not merge dict with {type(second)}!")

    return first


def compile_regex_pattern(pattern):
    try:
        res = re.compile(pattern)
    except re.error:
        res = None
    finally:
        return res


def as_namedtuple(d: dict, name=""):
    return collections.namedtuple(name, d.keys())(d.values())


def first_not_empty(*args):
    return next(x for x in args if len(x) > 0)


