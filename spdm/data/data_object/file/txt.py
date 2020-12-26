'''IO Plugin of Txt '''
import json

from spdm.util.utilities import as_file_fun
from spdm.util.logger import logger
__plugin_spec__ = {
    "name": "text",
    "filename_pattern": ["*.txt"],
    "support_data_type": [int, float, str, dict, list],
    "filename_extension": "txt"

}


@as_file_fun(mode="r")
def read(fid, *args, **kwargs):
    return fid.read()


@as_file_fun(mode="w")
def write(fid, d,  *args, **kwargs):
    fid.write(d)
