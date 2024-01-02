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
        if self.__class__ is Actor and self.__class__.__dispatch_init__(None, self, *args, **kwargs) is not False:
            return

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
        parent = getattr(self, "_parent", _not_found_)

        while isinstance(parent, AoS) and parent is not _not_found_:
            parent = getattr(parent, "_parent", _not_found_)

        # 尝试从父节点获得 inputs
        for name, edge in self.inputs.items():
           
            edge.source.update(getattr(parent, name, _not_found_))

    @property
    def tag(self) -> str:
        return f"{self._plugin_prefix}{self.__class__.__name__.lower()}"

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @property
    def MPI(self):
        return SP_MPI

    @contextlib.contextmanager
    def working_dir(self, suffix: str = "", prefix="") -> typing.Generator[pathlib.Path, None, None]:
        pwd = pathlib.Path.cwd()

        working_dir = f"{self.output_dir}/{prefix}{self.tag}{suffix}"

        temp_dir = None

        if SP_DEBUG:
            current_dir = pathlib.Path(working_dir)
            current_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix=self.tag)
            current_dir = pathlib.Path(temp_dir.name)

        os.chdir(current_dir)

        logger.info(f"Enter directory {current_dir}")

        try:
            yield current_dir
        except Exception as error:
            if temp_dir is not None:
                shutil.copytree(temp_dir.name, working_dir, dirs_exist_ok=True)
            os.chdir(pwd)
            logger.info(f"Enter directory {pwd}")
            logger.exception(f"Failed to execute actor {self.tag}! \n See log in {working_dir} ")
        else:
            if temp_dir is not None:
                temp_dir.cleanup()

            os.chdir(pwd)
            logger.info(f"Enter directory {pwd}")

    @property
    def output_dir(self) -> str:
        return (
            self.get("output_dir", None)
            or os.getenv("SP_OUTPUT_DIR", None)
            or f"{os.getcwd()}/{SP_LABEL.lower()}_output"
        )

    @property
    def current(self) -> TimeSlice:
        """当前时间片，指向 Actor 所在时间点的状态。"""
        return self.time_slice.current

    @property
    def previous(self) -> TimeSlice:
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

    def preprocess(self, *args, **kwargs) -> typing.Type[TimeSlice]:
        """Actor 的预处理，若需要，可以在此处更新 Actor 的状态树。"""
        self.time_slice.refresh(*args, **kwargs)
        return self.time_slice.current

    def execute(self, current: typing.Type[TimeSlice], *previous: typing.Type[TimeSlice]) -> typing.Type[TimeSlice]:
        """根据 inputs 和 前序 time slice 更新当前time slice"""
        return current

    def postprocess(self, current: typing.Type[TimeSlice]) -> typing.Type[TimeSlice]:
        """Actor 的后处理，若需要，可以在此处更新 Actor 的状态树。
        @param current: 当前时间片
        @param working_dir: 工作目录
        """
        pass

    def refresh(self, *args, **kwargs) -> typing.Type[TimeSlice]:
        """更新当前 Actor 的状态。
        更新当前状态树 （time_slice），并执行 self.iteration+=1

        """

        kwargs = self.inputs.update(kwargs)  # 更新 inputs，返回将不是 HTreeNode 的 input

        current = self.preprocess(*args, **kwargs)

        if current is None:
            current = self.time_slice.current

        current = self.execute(current, *self.time_slice.previous)

        if current is None:
            current = self.time_slice.current

        current = self.postprocess(current)

        if current is None:
            current = self.time_slice.current

        return current

    def advance(self, *args, dt: float | None = None, time: float | None = None, **kwargs) -> typing.Type[TimeSlice]:
        """推进 Actor 到下一时间片，向 time_slice 队列中压入新的时间片。"""

        # 保存当前状态
        self.flush()

        # 确定新的时间戳
        if time is None and dt is None:
            raise RuntimeError("time and dt are both None, do nothing")
        elif time is None and dt is not None:
            time = self.time + dt
        elif time is not None and time <= self.time:
            raise RuntimeError(f"time={time} is less than current time={self.time}, do nothing")
        elif dt is not None:
            logger.warning(f"ignore dt={dt} when time={time} is given")

        # 获得新的时间片
        new_slice = self.time_slice.advance(*args, time=time, **kwargs)

        return new_slice

    def flush(self, *args, **kwargs) -> typing.Type[TimeSlice]:
        """保存当前时间片的状态。
        根据当前 inputs 的状态，更新状态并写入 time_slice，
        默认 do nothing， 返回当前时间片
        """
        current = self.time_slice.current
        current.update(*args, **kwargs)
        return current
