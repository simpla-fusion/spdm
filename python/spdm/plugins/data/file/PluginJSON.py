import collections
import json

import numpy as np
from spdm.data.Entry import Entry
from spdm.data.File import FileHandler
from spdm.util.dict_util import as_native
from spdm.util.logger import logger


class JSONFile(FileHandler):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = self.path
        mode = self.mode_str
        try:
            self._fid = open(path,  mode=mode)
        except OSError as error:
            raise FileExistsError(f"Can not open file {path}! {error}")
        else:
            logger.debug(f"Open HDF5 File {path} mode={mode}")

    def read(self, *args,   **kwargs) -> Entry:
        return Entry(json.load(self._fid))

    def write(self,   d, *args,  **kwargs):
        json.dump(as_native(d, enable_ndarray=False), self._fid)


__SP_EXPORT__ = JSONFile
