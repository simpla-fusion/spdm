from __future__ import annotations

import tempfile
import shutil
import pathlib
import os
import typing
import numpy as np
import uuid
import contextlib
import inspect
from ..utils.logger import logger
from ..utils.plugin import Pluggable
from ..utils.envs import SP_MPI, SP_DEBUG, SP_LABEL
from ..utils.tags import _not_found_
from ..view import View as sp_view

from .HTree import HTreeNode
from .sp_property import SpTree, sp_property, sp_tree
from .Expression import Expression
from .TimeSeries import TimeSeriesAoS, TimeSlice
from .Path import update_tree
from .Edge import InPorts, OutPorts


@sp_tree
class Actor(Pluggable):
    mpi_enabled = False

    def __init__(self, *args, **kwargs) -> None:
        Pluggable.__init__(self, *args, **kwargs)
        SpTree.__init__(self, *args, **kwargs)
        self._uid = uuid.uuid3(uuid.uuid1(clock_seq=0), self.__class__.__name__)

        self._inputs = InPorts(self)
        self._outputs = OutPorts(self)

        tp_hints = typing.get_type_hints(self.__class__.refresh)
        for name, tp in tp_hints.items():
            if name == "return":
                continue
            elif getattr(tp, "_name", None) == "Optional":  # check typing.Optional
                t_args = typing.get_args(tp)
                if len(t_args) == 2 and t_args[1] is type(None):
                    tp = t_args[0]

            self._inputs[name].source.update(None, tp)

    @property
    def tag(self) -> str:
        return f"{self._plugin_prefix}{self.__class__.__name__.lower()}"

    @property
    def uid(self) -> int:
        return self._uid

    @property
    def MPI(self):
        return SP_MPI

    @contextlib.contextmanager
    def working_dir(self, suffix: str = "", prefix="") -> str:
        temp_dir = None
        if SP_DEBUG:
            _working_dir = f"{self.output_dir}/{prefix}{self.tag}{suffix}"
            pathlib.Path(_working_dir).mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix=self.tag)
            _working_dir = temp_dir.name

        pwd = os.getcwd()

        os.chdir(_working_dir)

        logger.info(f"Enter directory {_working_dir}")

        error = None

        try:
            yield _working_dir
        except Exception as e:
            error = e

        if error is not None and temp_dir is not None:
            shutil.copytree(temp_dir.name, f"{self.output_dir}/{self.tag}{suffix}", dirs_exist_ok=True)
        elif temp_dir is not None:
            temp_dir.cleanup()

        os.chdir(pwd)
        logger.info(f"Enter directory {pwd}")

        if error is not None:
            raise RuntimeError(
                f"Failed to execute actor {self.tag}! see log in {self.output_dir}/{self.tag}"
            ) from error

    @property
    def output_dir(self) -> str:
        return (
            self.get("output_dir", None)
            or os.getenv("SP_OUTPUT_DIR", None)
            or f"{os.getcwd()}/{SP_LABEL.lower()}_output"
        )

    @property
    def current(self) -> typing.Type[TimeSlice]:
        return self.time_slice.current

    @property
    def previous(self) -> typing.Type[TimeSlice]:
        return self.time_slice.previous

    time_slice: TimeSeriesAoS[TimeSlice]

    @property
    def time(self) -> float:
        """时间戳，代表 Actor 所处时间，用以同步"""
        return self.time_slice.current.time

    @property
    def iteration(self) -> int:
        return self.time_slice.current.iteration

    @property
    def inputs(self) -> InPorts:
        return self._inputs

    @property
    def outputs(self) -> OutPorts:
        return self._outputs

    def preprocess(self, *args, dt=None, time=None, **kwargs):
        if time is None and dt is None:
            pass
        elif time is not None and dt is not None:
            logger.warning(f"ignore dt={dt} when time={time} is given")
        elif time is None and dt is not None:
            time = self.time + dt

        if time is None and self._parent is not None:
            time = self._parent.time

        self._inputs.update(kwargs)

        self.time_slice.refresh(*args, time=time)

    def execute(self, current: TimeSlice, *previous_slices: typing.Tuple[TimeSlice], **kwargs) -> typing.Type[Actor]:
        """根据 inputs 和 前序 time slice 更显当前time slice"""
        pass

    def postprocess(self, current: TimeSlice):
        pass

    def refresh(self, *args, **kwargs) -> None:
        self.preprocess(*args, **kwargs)

        self.execute(self.time_slice.current, self.time_slice.previous, **self.inputs.fetch())

        self.postprocess(self.time_slice.current)

    def fetch(self, *args, slice_index=0, **kwargs) -> typing.Type[TimeSlice]:
        """
        获取 Actor 的输出
        """
        t = self.time_slice.get(slice_index)
        if not isinstance(t, SpTree):
            return t
        else:
            return t.clone(*args, **kwargs)
