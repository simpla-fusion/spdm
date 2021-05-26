import collections
from dataclasses import dataclass, fields, is_dataclass
from typing import (Any, Deque, Generic, Iterator, Mapping, NewType, Optional,
                    Sequence, TypeVar)


from ..data.Entry import Entry
from ..data.Node import Dict, List, Node, _TObject
from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.utilities import (_empty, _not_defined_, _not_found_,
                              guess_class_name)
from .Session import Session


class Actor(Dict[Node]):
    """
        Action/Event: Objects whose state changes over time
    """
    @dataclass
    class State:
        time: float

        def update(self, *args, **kwargs):
            pass

    def __new__(cls, desc=None, *args, **kwargs):
        prefix = getattr(cls, "_actor_module_prefix", None)
        n_cls = cls
        cls_name = None
        if cls is not Actor and prefix is None:
            pass
        elif isinstance(desc, collections.abc.Mapping):
            cls_name = desc.get("code", {}).get("name", None)
        elif isinstance(desc, Entry):
            cls_name = desc.get("code.name", _not_defined_)

        if isinstance(cls_name, str):
            try:
                n_cls = sp_find_module(f"{prefix}{cls_name}")
            except Exception:
                logger.error(f"Can not find actor '{prefix}{cls_name}'!")
                raise ModuleNotFoundError(f"{prefix}{cls_name}")
            else:
                logger.info(f"Load actor '{prefix}{cls_name}={guess_class_name(n_cls)}'!")

        return object.__new__(n_cls)

    def __init__(self, d=None,  *args, time: Optional[float] = None, maxlen: Optional[int] = None, dumper=None, **kwargs) -> None:
        super().__init__(d, *args, **kwargs)
        self._time = time if time is not None else 0.0
        self._job_id = 0  # Session.current().job_id(self.__class__.__name__)
        self._s_entry = dumper
        self._s_deque = collections.deque(maxlen=maxlen)

    @property
    def time(self):
        return 0.0  # self._time

    def job_id(self):
        return self._job_id

    def states(self) -> Sequence[State]:
        return self._s_deque

    def previous_state(self) -> State:
        return self._s_deque[-1]

    def current_state(self) -> State:
        """
            Function:  gather current state based on the dataclass ‘State’
            Return  :  state
        """
        assert(is_dataclass(self.State))

        return self.State(**collections.ChainMap({"time": self._time},
                                                 {f.name: getattr(self, f.name, _not_found_) for f in fields(self.__class__.State)}))

    def flush(self) -> State:
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

    def advance(self, *args, time: float = None, dt: float = None,   **kwargs) -> float:
        """
            Function: Advance the state of the Actor to the next time step. current -> next
            Return  : return the residual between the updated state and the previous state
                1. push current state to deque
                2. update current state
        """
        self.flush()
        if time is None:
            time = self.time+(dt or 1.0)
        logger.info(f"Advance actor to {self.time}. '{guess_class_name(self)}' ")
        return self.update(*args, time=time, **kwargs)

    def update(self, state: Optional[Mapping] = None, *args,   force=False, ** kwargs) -> float:
        """
            Function: update the current state of the Actor without advancing the time.
            Return  : return the residual between the updated state and the previous state
        """
        if state is not None:
            super().update(state)
        # super().__reset__({f.name: d.get(f.name, _not_found_) for f in fields(self.State) if f.name in d})

        self._time = self["time"]

        logger.info(f"Update actor at time={self.time}. '{guess_class_name(self)}'")

        return 0.0 if force else 0.0
