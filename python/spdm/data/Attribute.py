import numpy as np
import collections


class Attribute(np.ndarray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set(self, value):
        return self.copy(np.asarray(value))

    def get(self, value):
        return self.view(np.ndarray)


class AttributeCollection(collections.OrderedDict):
    def __init__(self,  *args, **kwargs):
        # super().__init__(*args, **kwargs)
        pass

    def __missing__(self, k):
        return self.setdefault(k,  Attribute(name=k))

    def __setitem__(self, k, v) -> None:
        self.__getitem__(k).set(v)
