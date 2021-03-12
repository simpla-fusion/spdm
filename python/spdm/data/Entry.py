
import collections
import pprint
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger


class Entry(object):

    def __init__(self, data=None,  *args, prefix=None, parent=None,   **kwargs):
        super().__init__()
        self._data = data
        self._parent = parent

        if not prefix:
            self._prefix = []
        elif isinstance(prefix, str):
            self._prefix = prefix.split(".")
        elif not isinstance(prefix, collections.abc.Sequence):
            self._prefix = [prefix]
        else:
            self._prefix = prefix

    def __repr__(self) -> str:
        return pprint.pformat(self.get_value([]))

    @property
    def lazy(self):
        return LazyProxy(self,  handler=self.__class__)

    @property
    def data(self):
        return self._data

    @property
    def parent(self):
        return self._parent

    @property
    def prefix(self):
        return self._prefix

    def child(self, path, *args, **kwargs):
        if not path:
            return self
        else:
            return self.__class__(self._data, prefix=self._normalize_path(path), parent=self._parent)

    def copy(self, other):
        # if isinstance(other, LazyProxy):
        #     other = other.__real_value__()
        # el
        if isinstance(other, Entry):
            other = other.entry.__real_value__()

        if isinstance(other, collections.abc.Mapping):
            for k, v in other.items():
                self._data[k] = v
        elif isinstance(other, collections.abc.Sequence):
            self._data.extend(other)
        else:
            raise ValueError(f"Can not copy {type(other)}!")

    def _normalize_path(self, path):
        if isinstance(path, str):
            path = path.split(".")
        elif not isinstance(path, collections.abc.Sequence):
            path = [path]
        return self._prefix + path

    def get(self, path=[], *args, **kwargs):
        path = self._normalize_path(path)

        obj = self._data

        for p in path:
            if type(p) is str and hasattr(obj, p):
                obj = getattr(obj, p)
            elif obj is not None:
                try:
                    obj = obj[p]
                except IndexError:
                    raise KeyError(path)
                except TypeError:
                    raise KeyError(path)
            else:
                raise KeyError(path)

        return obj

    def get_value(self,  path=[], *args, **kwargs):
        return self.get(path, *args, **kwargs)

    def put(self,  path, value, *args, **kwargs):
        path = self._normalize_path(path)

        obj = self._data

        if len(path) == 0:
            return obj
        for p in path[:-1]:
            if type(p) is str and hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                try:
                    t = obj[p]
                except KeyError:
                    obj[p] = {}
                    obj = obj[p]
                except IndexError as error:
                    raise IndexError(f"{p} > {len(obj)}! {error}")
                else:
                    obj = t
            # elif type(p) is int and p < len(obj):
            #     obj = obj[p]

            # else:
            #     obj[p] = {}
            #     obj = obj[p]
        if hasattr(obj, path[-1]):
            setattr(obj, path[-1], value)
        else:
            obj[path[-1]] = value

        return obj[path[-1]]

    def insert(self, path, v, *args, **kwargs):
        path = self._normalize_path(path)
        # FIXME: self._normalize_path(path)
        try:
            parent = self.get(path[:-1])
        except KeyError:
            parent = None
        idx = path[-1]
        if parent is not None and idx in parent:
            pass
        elif parent is not None and idx not in parent:
            parent[idx] = v
        elif isinstance(idx, str):
            parent = self.put(path[:-1], {})
            parent[idx] = v
        elif type(path[-1]) is int and path[-1] <= 0:
            parent = self.put(path[:-1], [])
            idx = 0
            parent[idx] = v
        return parent[idx]

    def update(self, path, v, *args, **kwargs):
        raise NotImplementedError()

    def delete(self, path=[], *args, **kwargs):
        path = self._normalize_path(path)

        if len(path) > 1:
            obj = self.get(path[:-1], *args, **kwargs)
        else:
            obj = self._data
        if hasattr(obj, path[-1]):
            delattr(obj, path[-1])
        else:
            del obj[path[-1]]

    def count(self,  path=[], *args, **kwargs):
        res = self.get(path, *args, **kwargs)
        if isinstance(res, (list, collections.abc.Mapping)):
            return len(res)
        else:
            return 1

    def contains(self, path, v, *args, **kwargs):
        obj = self.get(path, *args, **kwargs)
        return v in obj

    def call(self,   path=[], *args, **kwargs):
        obj = self.get(path)
        if callable(obj):
            res = obj(*args, **kwargs)
        elif len(args)+len(kwargs) == 0:
            res = obj
        else:
            raise TypeError(f"'{type(obj)}' is not callable")

        return res

    def push_back(self, path, v=None):
        parent = self.insert(path, [])
        parent.append(v or {})
        return path+[len(parent)-1]

    def pop_back(self, path):
        obj = self.get(path)
        res = None
        if obj is None:
            pass
        elif isinstance(obj, collections.abc.Sequence):
            res = obj[-1]
            obj.pop()
        else:
            raise KeyError(path)

        return res

    def iter(self, path=[], *args, **kwargs):
        yield from self.get(path, *args, **kwargs)

    def __iter__(self):
        yield from self.get()

    def __pre_process__(self, request, *args, **kwargs):
        return request

    def __post_process__(self, request, *args, **kwargs):
        if isinstance(request,  collections.abc.Sequence) and not isinstance(request, str):
            res = [self.__post_process__(v, *args, **kwargs) for v in request]
        elif isinstance(request, collections.abc.Mapping):
            res = {k: self.__post_process__(v, *args, **kwargs) for k, v in request.items()}
        else:
            res = request

        return res

# def is_entry(obj):
#     return isinstance(obj, LazyProxy) and isinstance(obj.__object__, Entry)
