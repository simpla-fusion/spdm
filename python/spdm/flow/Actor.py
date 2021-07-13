import collections.abc
from dataclasses import dataclass, fields, is_dataclass
from typing import (Any, Deque, Generic, Iterator, Mapping, NewType, Optional,
                    Sequence, TypeVar)


from ..data.Entry import Entry
from ..data.Node import Dict, List, Node, _TObject
from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.utilities import (_empty, _undefined_, _not_found_,
                              guess_class_name)
from .Session import Session

_TState = TypeVar("_TState")


class Actor(Dict[Node], Generic[_TState]):
    """
        Action/Event: Objects whose state changes over time
    """

    def __new__(cls, desc=None, *args, **kwargs):

        prefix = getattr(cls, "_actor_module_prefix", None)
        n_cls = cls
        cls_name = None
        if cls is not Actor and prefix is None:
            pass
        elif isinstance(desc, collections.abc.Mapping):
            cls_name = desc.get("code", {}).get("name", None)
        elif isinstance(desc, Entry):
            cls_name = desc.get("code.name", _undefined_)

        if isinstance(cls_name, str):
            n_cls = sp_find_module(f"{prefix}{cls_name}")

            if n_cls is None:
                logger.error(f"Can not find actor '{prefix}{cls_name}'! ")
                raise ModuleNotFoundError(f"{prefix}{cls_name}")
            else:
                logger.info(f"Load actor '{prefix}{cls_name}={guess_class_name(n_cls)}'!")

        return super(Actor, n_cls).__new__(n_cls, desc, *args, **kwargs)

    def __init__(self, d=None,  /, time: Optional[float] = None, maxlen: Optional[int] = None, dumper=None, **kwargs) -> None:

        super().__init__(d,  **kwargs)

        # logger.debug(f"Inititalize Actor {guess_class_name(self.__class__)}")

        self._time = time if time is not None else 0.0
        self._job_id = 0  # Session.current().job_id(self.__class__.__name__)
        self._s_entry = dumper
        self._s_deque = collections.deque(maxlen=maxlen)

    def __del__(self):
        # logger.debug(f"Delete Actor {guess_class_name(self.__class__)}")
        pass

    @property
    def time(self):
        return 0.0  # self._time

    def job_id(self):
        return self._job_id

    def states(self) -> Sequence[_TState]:
        return self._s_deque

    @property
    def previous_state(self) -> _TState:
        return self._s_deque[-1] if len(self._s_deque) > 0 else self

    def current_state(self) -> _TState:
        """
            Function:  gather current state based on the dataclass ‘State’
            Return  :  state
        """

        return collections.ChainMap({"time": self._time},
                                    {f.name: getattr(self, f.name, _not_found_) for f in fields(self.__class__.State)})

    def flush(self) -> _TState:
        current_state = self.current_state
        if self._s_entry is not None:
            next(self._s_entry).__reset__(current_state)

        self._s_deque.append(current_state)
        return current_state

    def rollback(self) -> bool:
        """
            Function : Roll back to the previous state
            Return   : if success return True
        """
        if self._s_deque.count() == 0 and self._s_entry is not None:
            self._s_deque.append(self._s_entry.fetch())

        return self.update(self._s_deque.pop(), force=True)

    def advance(self, *args, time: float = None, dt: float = None, update=False,  **kwargs) -> float:
        """
            Function: Advance the state of the Actor to the next time step. current -> next
            Return  : return the residual between the updated state and the previous state
                1. push current state to deque
                2. update current state
        """
        self.flush()

        if time is None:
            time = self.time+(dt or 1.0)

        logger.debug(f"Advance actor '{guess_class_name(self)}' to {self.time}.  ")

        if update:
            self.update(*args, time=time, **kwargs)

        return time

    def refresh(self,   *args, ** kwargs):
        """
            Function: update the current state of the Actor without advancing the time.
            Return  : return the residual between the updated state and the previous state
        """
        logger.debug(f"Refresh Actor {self.__class__}")
        return

    def reset(self, value=None, /, **kwargs) -> None:
        super().reset()
        if value is not None:
            self.update(value, **kwargs)
