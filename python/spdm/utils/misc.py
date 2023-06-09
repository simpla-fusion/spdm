import collections
import collections.abc
import dataclasses
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
import typing
from enum import Flag, auto
from typing import Sequence, Type
from urllib.parse import ParseResult, urlparse

import numpy as np
import yaml

from .tags import _empty, _not_found_, _undefined_
from .logger import logger


def float_unique(d: np.ndarray, x_min=-np.inf, x_max=np.inf) -> np.ndarray:
    if not isinstance(d, np.ndarray):
        d = np.asarray(d)
    d = np.sort(d)
    tag = np.append(True, np.diff(d)) > np.finfo(float).eps*10
    rtag = np.logical_and(d >= x_min, d <= x_max)
    rtag = np.logical_or(rtag, np.isclose(d, x_min, rtol=1e-8))
    rtag = np.logical_or(rtag, np.isclose(d, x_max, rtol=1e-8))
    tag = np.logical_and(tag, rtag)
    return d[tag]


def array_like(x: np.ndarray, d):
    if isinstance(d, np.ndarray):
        return d
    elif isinstance(d, (int, float)):
        return np.full_like(x, d)
    elif callable(d):
        return np.asarray(d(x))
    else:
        return np.zeros_like(x)
    # else:
    #     raise TypeError(type(d))


# _empty = object()


# class _NOT_FOUND_:
#     def _as_dict(self):
#         return {}

#     def _as_list(self):
#         return []

#     def __serialize__(self):
#         return None


# class _tags(Flag):
#     not_found = auto()
#     undefined = auto()


# _not_found_ = _tags.not_found

# _undefined_ = _tags.undefined


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


def try_get(obj, path: str, default_value=_undefined_):
    if obj is None or obj is _not_found_:
        return default_value
    elif path is None or path == '':
        return obj

    start = 0
    path = path.strip(".")
    s_len = len(path)
    while start >= 0 and start < s_len:
        pos = path.find('.', start)
        if pos < 0:
            pos = s_len
        next_obj = getattr(obj, path[start: pos], _not_found_)

        if next_obj is not _not_found_:
            obj = next_obj
        elif isinstance(obj, collections.abc.Mapping):
            next_obj = obj.get(path[start: pos], _not_found_)
            if next_obj is not _not_found_:
                obj = next_obj
            else:
                break
        else:
            break

        start = pos+1
    if start > s_len:
        return obj
    elif default_value is _undefined_:
        raise KeyError(f"Can for find path {path}")
    else:
        return default_value


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


# def try_get(holder, path, default_value=None):
#     path = normalize_path(path)
#     obj = holder

#     for k in path:
#         if isinstance(k, str):
#             next_o = getattr(obj, k, _not_found_)

#             op = getattr(obj.__class__, k, None)
#         else:
#             op = None
#         if op is None:
#             try:
#                 data = obj.__getitem__(k)
#             except Exception:
#                 obj = default_value
#                 break
#             # except IndexError as error:
#             #     raise IndexError(f"{k} > {len(obj)} Error: {error}")
#             else:
#                 obj = data

#         elif isinstance(op, functools.cached_property):
#             obj = op.__get__(obj)
#         elif isinstance(obj, property):
#             obj = op(obj, "fget")(obj)
#         else:
#             obj = default_value

#     return obj


def try_put(holder, path,  value):
    res = None
    if isinstance(path, str):
        res = getattr(holder.__class__, path, None)
    if res is None:
        holder.__setitem__(path, value)
    elif isinstance(res, property):
        getattr(res, "fset")(holder, path, value)
    else:
        raise KeyError(f"Can not set attribute {value}!")

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


def normalize_path(path):
    if path is None:
        path = []
    elif isinstance(path, str):
        path = path.split('.')
    elif not isinstance(path, collections.abc.MutableSequence):
        path = [path]
    return path


def serialize(d):
    if isinstance(d, (int, float, str)):
        return d
    elif hasattr(d, "__array__"):  # numpy.ndarray like
        return d.__array__()
    elif hasattr(d.__class__, "__serialize__"):
        return d.__serialize__()
    elif dataclasses.is_dataclass(d):
        return {f.name: serialize(getattr(d, f.name)) for f in dataclasses.fields(d)}
    elif isinstance(d, collections.abc.Mapping):
        return {k: serialize(v) for k, v in d.items()}
    elif isinstance(d, collections.abc.Sequence):
        return [serialize(v) for v in d]
    else:
        # logger.warning(f"Can not serialize {d.__class__.__name__}!")
        return f"<{d.__class__.__name__}>NOT_SERIALIZABLE!</{d.__class__.__name__}>"
        # raise TypeError(f"Can not serialize {type(d)}!")


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


def get_username():
    return getlogin()


def _gusses_name(self, name_hint):
    count = sum(1 for k, v in self._graph.nodes.items()
                if k == name_hint or k.startswith(f"{name_hint}_"))
    if count > 0:
        name_hint = f"{name_hint}_{count}"
    return name_hint


def _try_insert(self, name_hint, node):
    return self.add_node(node, label=self._gusses_name(name_hint or node.__class__.__name__.lower()))


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


def as_dataclass(cls, value):
    if not dataclasses.is_dataclass(cls):
        raise TypeError(type(cls))
    elif isinstance(value, collections.abc.Mapping):
        value = cls(**{k: value.get(k, None) for k in cls.__dataclass_fields__})
    elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        value = cls(*value)
    else:
        raise TypeError(f"Can not convert '{type(value)}' to dataclass")
        # value = type_hint(value, **kwargs)
    return value


def guess_class_name(obj):

    if not inspect.isclass(obj):
        cls = obj.__class__
        cls_name = getattr(obj, "__orig_class__", None)
    else:
        cls = obj
        cls_name = None

    return cls_name or f"{cls.__module__}.{cls.__name__}"


def find_duplicate(l: Sequence, atol=1.0e-8):
    for i, val in enumerate(l[:-1]):
        idx = np.flatnonzero(np.abs(l[i+1:]-val) < atol)
        if len(idx) > 2:
            yield idx+i


def get_value_by_path(data, path, default_value=None):
    # 将路径按 '/' 分割成列表
    if isinstance(path, str):
        segments = path.split("/")
    elif isinstance(path, collections.abc.Sequence):
        segments = path
    else:
        segments = [path]

    # 初始化当前值为 data
    current_value = data
    # 遍历路径中的每一段
    for segment in segments:
        # 如果当前值是一个字典，并且包含该段作为键
        if isinstance(current_value, collections.abc.Mapping) and segment in current_value:
            # 更新当前值为该键对应的值
            current_value = current_value[segment]
        else:
            # 否则尝试将该段转换为整数索引
            try:
                index = int(segment)
                # 如果当前值是一个列表，并且索引在列表范围内
                if isinstance(current_value, list) and 0 <= index < len(current_value):
                    # 更新当前值为列表中对应索引位置的元素
                    current_value = current_value[index]
                else:
                    # 否则返回默认值
                    return default_value
            except ValueError:
                # 如果转换失败，则返回默认值
                return default_value
    # 返回最终的当前值
    return current_value


def set_value_by_path(data, path, value):
    # 将路径按 '/' 分割成列表
    segments = path.split("/")
    # 初始化当前字典为 data
    current_dict = data
    # 遍历路径中除了最后一段以外的每一段
    for segment in segments[:-1]:
        # 如果当前字典包含该段作为键，并且对应的值也是一个字典
        if segment in current_dict and isinstance(current_dict[segment], dict):
            # 更新当前字典为该键对应的子字典
            current_dict = current_dict[segment]
        else:
            # 尝试将该段转换为整数索引
            try:
                index = int(segment)
                # 如果当前字典不包含该段作为键，或者对应的值不是一个列表
                if segment not in current_dict or not isinstance(current_dict[segment], list):
                    # 创建一个空列表作为该键对应的值
                    current_dict[segment] = []
                # 更新当前字典为该键对应的子列表
                current_dict = current_dict[segment]
            except ValueError:
                # 如果转换失败，则抛出一个异常，提示无法继续查找
                raise Exception(f"Cannot find {segment} in {current_dict}")
    # 在当前字典中设置最后一段作为键，给定的值作为值
    last_segment = segments[-1]
    # 尝试将最后一段转换为整数索引
    try:
        index = int(last_segment)
        # 如果当前字典包含最后一段作为键，并且对应的值是一个列表
        if last_segment in current_dict and isinstance(current_dict[last_segment], list):
            # 判断索引是否在列表范围内
            if 0 <= index < len(current_dict[last_segment]):
                # 更新列表中对应索引位置的元素为给定值
                current_dict[last_segment][index] = value
            else:
                # 否则抛出一个异常，提示无法更新列表元素
                raise Exception(f"Index {index} out of range for list {current_dict[last_segment]}")
        else:
            # 否则直接设置最后一段作为键，给定值作为值（此时会创建一个单元素列表）
            current_dict[last_segment] = value
    except ValueError:
        # 如果转换失败，则直接设置最后一段作为键，给定值作为值（此时会覆盖原有列表）
        current_dict[last_segment] = value

    return True


def get_value(*args, **kwargs) -> typing.Any:
    return get_value_by_path(*args, **kwargs)


def get_many_value(d: collections.abc.Mapping, name_list: collections.abc.Sequence, default_value=None) -> collections.abc.Mapping:
    return {k: get_value(d, k, get_value(default_value, idx)) for idx, k in enumerate(name_list)}


def set_value(*args, **kwargs) -> bool:
    return set_value_by_path(*args, **kwargs)


def replace_tokens(value: str, env):
    class _TokenProxy:
        def __init__(self, d):
            if isinstance(d, _TokenProxy):
                self._data = d._data
            else:
                self._data = d

        def __getitem__(self, key):
            value = get_value_by_path(self._data, key.split('.'), _not_found_)
            if value is _not_found_:
                return '{' + key + '}'
            else:
                return value

    if isinstance(value, str):
        # 使用 format_map() 方法进行替换，并更新 document 中的 value
        return value.format_map(_TokenProxy(env))
    elif isinstance(value, list):
        return [replace_tokens(v, env) for v in value]
    elif isinstance(value, dict):
        return {k: replace_tokens(v, env) for k, v in value.items()}
    else:
        return value


def fetch_request(url: str) -> typing.Dict:
    """
        根据路径拖回并解析module_file
    """
    if not isinstance(url, ParseResult):
        url = urlparse(url)
    content = None

    if url.scheme in ['http', 'https']:
        # path is a uri
        response = requests.get(url)
        if response.status_code == 200:
            # request is successful
            content = yaml.safe_load(response.text)  # parse the response text as yaml
        else:
            raise FileNotFoundError(url)
    elif url.scheme in ['local', 'file', '']:
        if not os.path.isfile(url.path):
            raise FileNotFoundError(url.path)
        # file exists
        with open(url.path, 'r') as file:
            content = yaml.safe_load(file)  # parse the file content as yaml
    else:
        raise NotImplementedError(url)

    return content


primitive_types = (int, bool, str, float, complex, np.ndarray)

builtin_types = (int, bool, str, float, complex, list, dict, set, tuple, np.ndarray)


def typing_get_origin(tp):
    if tp is None or tp is _not_found_:
        return None
    elif isinstance(tp, (typing._GenericAlias, typing.GenericAlias)):
        return typing.get_origin(tp)
    elif not inspect.isclass(tp):
        return None

    orig_class = getattr(tp, "__orig_bases__", None)

    if orig_class is None:
        return tp
    elif isinstance(orig_class, tuple):
        return typing_get_origin(orig_class[0])
    else:
        return typing_get_origin(orig_class)


def group_dict_by_prefix(d: collections.abc.Mapping, prefixes: str | typing.List[str], keep_prefix=False, sep='_') -> typing.Tuple[typing.Dict, ...]:
    """ 将 字典 d 根据prefixs进行分组
     prefix 是一个字符串，或者是一个字符串列表
     key具有相同prefix的项会被分到一组， (if keep_prefix is True then prefix会被去掉)
     返回值是一个元组，每元素是分组后的字典，最后一个元素是其他项的字典
    """

    if isinstance(prefixes, str):
        prefixes = prefixes.split(',')

    groups = [None]*len(prefixes) + [{}]

    for key, value in d.items():
        for idx, prefix in enumerate(prefixes):

            if not key.startswith(prefix):
                continue

            if key == prefix:
                if groups[idx] is None:
                    groups[idx] = value
                elif isinstance(value, collections.abc.Mapping):
                    groups[idx].update(value)
                else:
                    raise ValueError(f"prefix {prefix} is not unique")
            else:
                if sep is not None:
                    prefix = prefix+sep

                if not key.startswith(prefix):
                    continue

                if not keep_prefix:
                    key = key[len(prefix):]

                if groups[idx] is None:
                    groups[idx] = {}
                elif not isinstance(groups[idx], collections.abc.Mapping):
                    raise ValueError(f"prefix {prefix} is not a dict, but {groups[idx]}")

                groups[idx][key] = value
            break
        else:
            groups[-1][key] = value
    return tuple(groups)


def fun():
    if isinstance(prefixs, str):
        prefixs = [prefixs]

    prefix_dict = {}

    if isinstance(obj, dict):
        prefix_dict = obj
        obj = None

    prefix_len = len(prefix)
    other_dict = {}
    for k, v in d.items():
        if k.startswith(prefix):
            if not keep_prefix:
                k = k[prefix_len:]
            if not isinstance(v, dict) or not isinstance(prefix_dict.get(k, None), dict):
                prefix_dict[k] = v
            else:
                prefix_dict[k] |= v

        elif k != prefix[:-1]:
            other_dict[k] = v

    if obj is None:
        return prefix_dict, other_dict
    else:
        if len(prefix_dict) > 0:
            raise RuntimeWarning(f" {prefix}*  {prefix_dict} are ignored")

        return obj, other_dict
