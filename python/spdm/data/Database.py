import collections
import contextlib
import pathlib
import shutil
import uuid

from ..util.logger import logger
from .Collection import Collection
from .Connection import Connection


class Database(Connection):

    """ 
    """

    def __init__(self,  *args,  **kwargs):
        super(Connection, self).__init__(*args,  ** kwargs)
        super(Collection, self).__init__()
