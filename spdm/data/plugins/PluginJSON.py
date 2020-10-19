import collections
import json

import numpy as np
from spdm.util.logger import logger

from ..Collection import FileCollection
from ..Document import Document


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        logger.debug(type(obj))
        return super().default(obj)


class JSONDocument(Document):
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._path = path

    def load(self, *args, path=None,  **kwargs):
        with open(path or self._path, mode="r") as fid:
            self.root._holder = json.load(fid)

    def save(self,   *args, path=None, **kwargs):
        with open(path or self._path, mode="w") as fid:
            json.dump(self.root._holder, fid, cls=NumpyEncoder)


class JSONCollection(FileCollection):
    def __init__(self, uri, *args, **kwargs):
        super().__init__(uri, *args,
                         file_extension=".json",
                         file_factory=lambda *a, **k: JSONDocument(*a, **k),
                         ** kwargs)
