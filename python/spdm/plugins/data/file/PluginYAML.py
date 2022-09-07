import collections
import yaml

import numpy
from spdm.logger import logger

from spdm.data.Collection import FileCollection
from spdm.data.Document import Document


class YAMLDocument(Document):
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.load(path, mode=self.mode)

    def load(self, *args, **kwargs):
        with self.open(mode="r") as fid:
            if fid is not None:
                self._holder = yaml.load(fid, Loader=yaml.CLoader)
            else:
                self._holder = None

    def save(self, d, *args, **kwargs):
        with self.open(mode="w") as fid:
            yaml.dump(self._holder, fid,  Dumper=yaml.CDumper)


class YAMLCollection(FileCollection):
    def __init__(self, uri, *args, **kwargs):
        super().__init__(uri, *args,
                         file_extension=".json",
                         file_factory=lambda *a, **k: YAMLDocument(*a, **k),
                         ** kwargs)
