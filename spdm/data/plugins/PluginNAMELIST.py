import collections
import f90nml

import numpy as np
from spdm.util.logger import logger

from ..Collection import FileCollection
from ..Document import Document


class NumpyEncoder(json.NAMELISTEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        logger.debug(type(obj))
        return super().default(obj)


class NAMELISTDocument(Document):
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._path = path

    def load(self, *args, path=None,  **kwargs):
        with open(path or self._path, mode="r") as fid:
            self.root._holder = f90nml.read(fid).todict(complex_tuple=True)

    def save(self,   *args, path=None, template=None, **kwargs):
        with open(path or self._path, mode="w") as fid:
            d = self._encode("", self.root._holder)
            if template is None:
                f90nml.write(d, fid)
            else:
                if isinstance(template, str):
                    template = urisplit(template)
                f90nml.patch(template.path, d, fid)

    def _encode(self, prefix, nobj):
        if isinstance(nobj, str):
            return nobj
        elif isinstance(nobj, collections.abc.Mapping):
            return {k: self._encode(f"{prefix}.{k}", p) for k, p in nobj.items()}
        elif isinstance(nobj, collections.abc.Sequence):
            return [self._encode(f"{prefix}.{k}", p) for k, p in enumerate(nobj)]
        elif type(nobj) not in [str, int, float, bool, type(None)]:
            return str(nobj)
        else:
            return nobj


class NAMELISTCollection(FileCollection):
    def __init__(self, uri, *args, **kwargs):
        super().__init__(uri, *args,
                         file_extension=".nml",
                         file_factory=lambda *a, **k: NAMELISTDocument(*a, **k),
                         ** kwargs)
