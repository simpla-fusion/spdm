import collections
import json

import numpy as np
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.util.dict_util import as_native
from spdm.common.logger import logger


class JSONFile(File):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open(self):
        path = self.path
        mode = self.mode_str
        try:
            self._fid = open(path,  mode=mode)
        except OSError as error:
            raise FileExistsError(f"Can not open file {path}! {error}")
        else:
            logger.debug(
                f"Open {self.__class__.__name__} File {path} mode={mode}")
        return self

    def read(self, *args,   **kwargs) -> Entry:
        if not hasattr(self, "_fid"):
            self.open()
        return Entry(json.load(self._fid))

    def write(self,   d, *args,  **kwargs):
        json.dump(as_native(d, enable_ndarray=False), self._fid)


__SP_EXPORT__ = JSONFile
