from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import collections


class Node(object):
    def __init__(self, holder=None, *args,   **kwargs):
        super().__init__()
        self._holder = holder if holder is not None else None

    @property
    def holder(self):
        return self._holder

    def get_entry(self):
        return LazyProxy(self, handler=self.__class__)

    def set_entry(self, other):
        return self.copy(other)

    entry = property(get_entry, set_entry, None, "entry point of the node")

    def copy(self, other):
        if isinstance(other, LazyProxy):
            other = other.__real_value__()
        elif isinstance(other, Node):
            other = other.entry.__real_value__()

        if isinstance(other, collections.abc.Mapping):
            for k, v in other.items():
                self._holder[k] = v
        elif isinstance(other, collections.abc.Sequence):
            self._holder.extend(other)
        else:
            raise ValueError(f"Can not copy {type(other)}!")

    def put(self,  path, value, *args, **kwargs):
        obj = self._holder
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

    def get(self, path, *args, **kwargs):
        obj = self._holder
        for p in path:
            if type(p) is str and hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                try:
                    obj = obj[p]
                except IndexError:
                    raise KeyError(path)
                except TypeError:
                    raise KeyError(path)
        return obj

    def insert(self, path, v, *args, **kwargs):
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

    def get_value(self,  path, *args, **kwargs):
        return self.get(path, *args, **kwargs)

    def delete(self, path, *args, **kwargs):
        if len(path) > 1:
            obj = self.get(path[:-1], *args, **kwargs)
        else:
            obj = self._holder
        if hasattr(obj, path[-1]):
            delattr(obj, path[-1])
        else:
            del obj[path[-1]]

    def count(self,  path, *args, **kwargs):
        obj = self.get(path, *args, **kwargs)
        return len(obj)

    def contains(self, path, v, *args, **kwargs):
        obj = self.get(path, *args, **kwargs)
        return v in obj

    def call(self,   path, *args, **kwargs):
        obj = self.get(path)
        return obj(*args, **kwargs)

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

    def iter(self, path, *args, **kwargs):
        yield from self.get(path, *args, **kwargs)
