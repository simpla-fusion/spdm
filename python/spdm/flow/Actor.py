from ..util.SpObject import SpObject
from ..util.logger import logger

from .Session import Session


class Actor(SpObject):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        if cls is not Actor:
            return object.__new__(cls)
        else:
            return SpObject.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._job_id = Session.current().job_id(self.__class__.__name__)

    @property
    def job_id(self):
        return self._job_id
