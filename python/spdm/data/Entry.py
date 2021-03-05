
import collections

from spdm.util.logger import logger

# from spdm.util.urilib import urisplit
# from .Collection import Collection
# from .Document import Document
# def open_entry(desc, *args, **kwargs):
#     if isinstance(desc, str):
#         desc = urisplit(desc)

#     # else:
#     #     raise TypeError(f"Illegal uri type! {desc}")
#     if kwargs is not None and len(kwargs) > 0:
#         desc[] = kwargs

#     if desc.get("schema", None) is not None:
#         holder = Collection(desc)
#     else:
#         holder = Document(desc)

#     if not desc.get("fragment", None):
#         return holder.entry
#     else:
#         return holder.open(**desc.fragment.__data__).entry

class Entry(object):
    def __init__(self, holder,  *args, parent=None, prefix=None, **kwargs):
        super().__init__()
        self._holder = holder
        self._parent = parent
        self._prefix = prefix or []

    # def __del__(self):
    #     try:
    #         self._holder=None
    #     except Exception as error:
    #         logger.error(error)

    @property
    def holder(self):
        return self._holder

    @property
    def parent(self):
        return self._parent

    @property
    def prefix(self):
        return self._prefix

    # @property
    # def entry(self):
    #     return LazyProxy(self,  handler=self.__class__)

    # @entry.setter
    # def entry(self, other):
    #     return self.copy(other)

    def copy(self, other):
        # if isinstance(other, LazyProxy):
        #     other = other.__real_value__()
        # el
        if isinstance(other, Entry):
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

    def iter(self, path, *args, **kwargs):
        yield from self.get(path, *args, **kwargs)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.put(key, value)


# def is_entry(obj):
#     return isinstance(obj, LazyProxy) and isinstance(obj.__object__, Entry)
