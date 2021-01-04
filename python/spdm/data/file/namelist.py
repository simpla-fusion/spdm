import collections
import copy
import pathlib
import pprint
import shutil
from typing import Any, Dict

import f90nml
from spdm.util.logger import logger
from spdm.util.dict_util import normalize_data
from ..File import File


class FileNamelist(File):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    def update(self, data, *args, **kwargs):
        data = normalize_data(data)
        logger.debug(data)
        f90nml.patch(self.template.as_posix(), data, self.path.as_posix())

    def read(self, *args, **kwargs) -> Dict[str, Any]:
        return f90nml.read(self.path.open(mode="r")).todict(complex_tuple=True)

    # def normalize_r(self, prefix, nobj):
    #     if isinstance(nobj, str):
    #         return nobj
    #     elif isinstance(nobj, collections.abc.Mapping):
    #         return {k: self.normalize_r(f"{prefix}.{k}", p) for k, p in nobj.items()}
    #     elif isinstance(nobj, collections.abc.Sequence):
    #         return [self.normalize_r(f"{prefix}.{k}", p) for k, p in enumerate(nobj)]
    #     elif type(nobj) not in [str, int, float, bool, type(None)]:
    #         return str(nobj)
    #     else:
    #         return nobj

    def write(self,  data: Dict[str, Any], *args,  **kwargs):
        f90nml.write(normalize_data(data), self.path.open(mode="w"))
        # d = d or {}

        # if isinstance(fid, pathlib.Path):
        #     fid = fid.as_posix()

        # d = self.normalize_r("", d)

        # if template is None:
        #     f90nml.write(d, fid)
        # else:
        #     if not isinstance(template, SpURI):
        #         template = SpURI(template)
        #     f90nml.patch(template.path, d, fid)
        pass


__SP_EXPORT__ = FileNamelist
