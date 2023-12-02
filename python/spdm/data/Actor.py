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
from .AoS import AoS


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

        # 查找父节点的输入
        parent = self._parent
        while isinstance(parent, AoS) and parent is not _not_found_:
            parent = getattr(parent, "_parent", _not_found_)

        p_inputs = getattr(parent, "inputs", _not_found_)
        if isinstance(p_inputs, InPorts):
            # 尝试从父节点获得 inputs
            for name, edge in self.inputs.items():
                if edge.source.node is not None:
                    continue
                node = p_inputs.get_source(name, _not_found_)
                
                if node is _not_found_:
                    node = getattr(parent, name, _not_found_)

                if node is not _not_found_:
                    edge.source.update(node)

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
        """当前时间片，指向 Actor 所在时间点的状态。"""
        return self.time_slice.current

    @property
    def previous(self) -> typing.Type[TimeSlice]:
        """前一个时间片，指向 Actor 在前一个时间点的状态。"""
        return self.time_slice.previous

    time_slice: TimeSeriesAoS[TimeSlice]
    """ 时间片序列，保存 Actor 历史状态。
        @note: TimeSeriesAoS 长度为 n(=3) 循环队列。当压入序列的 TimeSlice 数量超出 n 时，会调用 TimeSeriesAoS.__full__(first_slice)  
    """

    @property
    def time(self) -> float:
        """当前时间，"""
        return self.time_slice.current.time

    @property
    def iteration(self) -> int:
        """当前时间片执行 refresh 的次数。对于新创建的 TimeSlice，iteration=0"""
        return self.time_slice.current.iteration

    @property
    def inputs(self) -> InPorts:
        """保存输入的 Edge，记录对其他 Actor 的依赖。"""
        return self._inputs

    @property
    def outputs(self) -> OutPorts:
        """保存外链的 Edge，可视为对于引用（reference）的记录"""
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

    def refresh(self, *args, time=None, **kwargs) -> None:
        """更新当前 Actor 的状态。
        若 time 为 None 或者与当前时间一致，则更新当前状态树，并执行 self.iteration+=1
        否则，向 time_slice 队列中压入新的时间片。
        """
        self.preprocess(*args, time=time, **kwargs)

        self.execute(self.time_slice.current, self.time_slice.previous, **self.inputs.fetch())

        self.postprocess(self.time_slice.current)

    def advance(self, *args, dt: float = None, time: float = None, **kwargs) -> None:
        if time is None and dt is None:
            raise RuntimeError("time and dt are both None, do nothing")
        elif time is None:
            time = self.time + dt
        elif time <= self.time:
            raise RuntimeError(f"time={time} is less than current time={self.time}, do nothing")
        elif dt is not None:
            logger.warning(f"ignore dt={dt} when time={time} is given")

        return self.refresh(*args, time=time, **kwargs)

    def fetch(self, *args, slice_index=0, **kwargs) -> typing.Type[TimeSlice]:
        """
        获取 Actor 的输出
        """
        t = self.time_slice.get(slice_index)
        if not isinstance(t, SpTree):
            return t
        else:
            return t.clone(*args, **kwargs)
