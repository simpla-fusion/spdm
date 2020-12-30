import collections
import copy
import pathlib
import pprint
import shutil
from typing import Any, Dict

import f90nml
from spdm.util.logger import logger

from ..File import File

__plugin_spec__ = {
    "name": "namelist",
    "filename_pattern": ["*.nml"],
    "filename_extension": "nml",
    "support_data_type": [int, float, str, dict]
}


class FileNamelist(File):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        logger.debug(data)

    def read(self, fid, *args, **kwargs) -> Dict[str, Any]:
        return f90nml.read(fid).todict(complex_tuple=True)

    def normalize_r(self, prefix, nobj):
        if isinstance(nobj, str):
            return nobj
        elif isinstance(nobj, collections.abc.Mapping):
            return {k: self.normalize_r(f"{prefix}.{k}", p) for k, p in nobj.items()}
        elif isinstance(nobj, collections.abc.Sequence):
            return [self.normalize_r(f"{prefix}.{k}", p) for k, p in enumerate(nobj)]
        elif type(nobj) not in [str, int, float, bool, type(None)]:
            return str(nobj)
        else:
            return nobj

    def write(self,  d: Dict[str, Any], *args, template=None, **kwargs):
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
