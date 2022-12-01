import contextlib
import io
import pathlib
import shutil
import tempfile
import uuid

from ..util.logger import logger
from spdm.util.urilib import urisplit

from .DataObject import DataObject
from .Document import Document


class Directory(DataObject):
    """ Default entry for file-like object
    """

    def __init__(self,  desc, value=None, *args, **kwargs):
        pass
