import collections
from dataclasses import dataclass, is_dataclass
from typing import Any, Deque, Generic, Mapping, NewType, Optional, TypeVar

import numpy as np

from ..data.Entry import Entry
from ..data.Node import Dict, List, Node, _TObject
from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.utilities import _empty, _not_found_, guess_class_name
from .Session import Session


class Actor(Dict[str, Node]):
    """
        Action/Event: Objects whose state changes over time
    """
    @dataclass
    class State:
        pass

    def __new__(cls, desc=None, *args, **kwargs):
        prefix = getattr(cls, "_actor_module_prefix", None)
        n_cls = cls
        if cls is not Actor and prefix is None:
            pass
        elif isinstance(desc, collections.abc.Mapping):
            name = desc.get("code", {}).get("name", None)
            if name is not None:
                try:
                    n_cls = sp_find_module(f"{prefix}{name}")
                except Exception:
                    logger.error(f"Can not find actor '{prefix}{name}'!")
                    raise ModuleNotFoundError(f"{prefix}{name}")
                else:
                    logger.info(f"Load actor '{guess_class_name(n_cls)}'!")

        #     if cls is not Actor:
        #         return object.__new__(cls)
        #     else:
        #         return SpObject.__new__(cls, *args, **kwargs)
        return object.__new__(n_cls)

    def __init__(self, entry: Optional[Entry] = None, *args, time: Optional[float] = None, maxlen: Optional[int] = None,  **kwargs) -> None:
        super().__init__(entry, *args, **kwargs)
        self._time = time if time is not None else 0.0
        self._job_id = 0  # Session.current().job_id(self.__class__.__name__)
        self._entry = entry
        self._s_deque = collections.deque(maxlen=maxlen)

    @property
    def job_id(self):
        return self._job_id

    @property
    def previous_state(self) -> State:
        return self._s_deque[-1]

    @property
    def current_state(self) -> State:
        """
            Function:  gather current state based on the dataclass ‘State’
            Return  :  state
        """
        cls_state = self.__class__.State
        assert(is_dataclass(cls_state))
        d = {}
        for k in dir(self.__class__.State.__annotations__):
            d[k] = getattr(self, k, _not_found_)
        return self.__class__.State(**collections.ChainMap(self._time, d))

    def advance(self, *args, time: float = None, dt: float = None,   **kwargs) -> float:
        """
            Function: Advance the state of the Actor to the next time step. current -> next
            Return  : return new time

                1. push current state to deque
                2. upate current state
        """
        if time is None:
            time = self._time[-1]+(dt or 1.0)
        self._time = time
        logger.info(f"Advance actor to {self._time[-1]}. '{guess_class_name(self)}' ")
        return self._time

    def rollback(self, n_step: int = 1) -> bool:
        """
            Function : Roll back to the previous state
            Return   : if success return True
        """

        return NotImplemented

    def update(self,  *args,  **kwargs) -> float:
        """
            Function: update the current state of the Actor without advancing the time.
            Return  : return the residual between the updated state and the previous state
        """
        logger.info(f"Update actor at time={self._time}. '{guess_class_name(self)}'")
        return 0.0
