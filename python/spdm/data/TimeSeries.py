
import typing

from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.utils.logger import logger


_T = typing.TypeVar("_T")



class TimeSlice(Dict[Node]):

    time: float = sp_property(unit='s',type='dynamic')
    


class TimeSeriesAoS(List[_T]):
    """
        A series of time slices, each time slice is a state of the system at a given time.
        Each slice is a dict .
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(self._appinfo)

    def update(self,  *args, dt=None, time=None, **kwargs):
        """
            update the last time slice, base on profiles_2d[-1].psi
        """

        pass
