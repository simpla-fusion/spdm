import collections
import os
import pathlib
import pprint
import shlex
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import List
from functools import cached_property

from ..data.DataObject import DataObject
from ..util.AttributeTree import AttributeTree
from ..util.dict_util import format_string_recursive
from ..util.logger import logger
from ..util.Signature import Signature
from ..util.SpObject import SpObject
from .Session import Session


class SpModule(SpObject):

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if cls is not SpModule:
            return object.__new__(cls)
        else:
            return SpObject.__new__(cls, *args, **kwargs)

    def __init__(self, *args, envs=None, metadata=None,  **kwargs):
        super().__init__(metadata=metadata)

        self._envs = envs or {}
        self._args = args
        self._kwargs = kwargs
        self._inputs = None
        self._outputs = None

        self._envs["JOB_ID"] = Session.current_job or self.__class__.__name__

    def __del__(self):
        super().__del__()

    @property
    def envs(self):
        return collections.ChainMap(self._envs, self._metadata)

    def preprocess(self):
        logger.debug(f"Preprocess: {self.__class__.__name__}")

    def postprocess(self):
        logger.debug(f"Postprocess: {self.__class__.__name__}")

    def execute(self):
        logger.debug(f"Execute: {self.__class__.__name__}")
        return None

    @cached_property
    def inputs(self):
        """
            Collect and convert inputs
        """

        args = DataObject.create(self._args or [])

        kwargs = DataObject.create(self._kwargs)

        envs = collections.ChainMap(kwargs, {"VAR_ARGS": args}, self.envs)

        for p_in in self.metadata.in_ports:

            p_name = str(p_in["name"])

            if p_name == "VAR_ARGS":
                args = DataObject.create(args, _metadata=format_string_recursive(p_in, envs))
            elif p_name in self._kwargs:
                kwargs[p_name] = DataObject.create(kwargs[p_name], _metadata=format_string_recursive(p_in, envs))
            else:
                kwargs[p_name] = DataObject.create(p_in.get("default", None),
                                                   _metadata=format_string_recursive(p_in, envs))

        logger.debug(kwargs)

        return args, kwargs

    @property
    def outputs(self):
        if not self._outputs:
            self._outputs = self.run()
            self._inputs = None

        return self._outputs

    def run(self):

        args, kwargs = self.inputs

        self.preprocess()

        error_msg = None

        try:
            logger.debug(f"Execute Start: {self.metadata.annotation.label}")
            res = self.execute(*args, **kwargs)
            logger.debug(f"Execute Done : {self.metadata.annotation.label}")
        except Exception as error:
            error_msg = error
            logger.error(f"Execute Error! {error}")
            res = None

        self.postprocess()

        if error_msg is not None:
            raise error_msg

        return res


class SpModuleLocal(SpModule):
    """Call subprocess/shell command
    {PKG_PREFIX}/bin/xgenray  
    """

    script_call = {
        ".py": sys.executable,
        ".sh": "bash",
        ".csh": "tcsh"
    }

    def __init__(self, *args, working_dir=None, **kwargs):
        super().__init__(*args, **kwargs)

        working_dir = working_dir or Session.current().cwd

        if isinstance(working_dir, str):
            working_dir = pathlib.Path(working_dir)

        count = len(list(working_dir.glob(f"{self.__class__.__name__}_*")))

        working_dir /= f"{self.__class__.__name__}_{count}"
        working_dir = working_dir.expanduser().resolve()

        working_dir.mkdir(exist_ok=True, parents=True)

        self._working_dir = working_dir

        self._envs["WORKING_DIR"] = working_dir

        logger.debug(f"Initialize: {self.__class__.__name__} at {self.working_dir} ")

    def __del__(self):
        logger.debug(f"Finalize: {self.__class__.__name__} ")

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def inputs(self):
        if self._inputs is not None:
            return self._inputs

        pwd = pathlib.Path.cwd()
        os.chdir(self.working_dir)
        res = super().inputs
        os.chdir(pwd)
        return res

    def _execute_module_command(self, *args):
        logger.debug(f"MODULE CMD: module {' '.join(args)}")
        py_commands = os.popen(f"{os.environ['LMOD_CMD']} python {' '.join(args)}  ").read()
        res = exec(py_commands)
        return res

    def _execute_process(self, cmd):
        res = os.popen(cmd).read()
        logger.debug(f"SHELL CMD: {cmd} : {res}")
        return res

    def _execute_object(self, cmd):
        return NotImplemented

    def _execute_script(self, cmds):
        if cmds is None:
            return None
        elif isinstance(cmds, collections.abc.Sequence) and not isinstance(cmds, str):
            pass
        else:
            cmds = [cmds]

        res = None

        for cmd in cmds:
            if isinstance(cmd, collections.abc.Mapping):
                res = self._execute_object(cmd)
            elif isinstance(cmd, str):
                if cmd.startswith("module "):
                    res = self._execute_module_command(cmd[len("module "):])
                elif not self._only_module_command:
                    res = self._execute_process(cmd)
                else:
                    raise RuntimeError(f"Illegal command! [{cmd}] Only 'module' command is allowed.")
            elif isinstance(cmd, collections.abc.Sequence):
                res = self._execute_script(cmd)
            elif not cmd:
                res = None
            else:
                raise NotImplementedError(cmd)

        return res

    def preprocess(self):
        super().preprocess()
        self._execute_script(self.metadata.prescript)

    def postprocess(self):
        self._execute_script(self.metadata.postscript)
        super().postprocess()

    def execute(self, *args, **kwargs):
        module_name = str(self.metadata.annotation.name)

        module_root = pathlib.Path(os.environ.get(f"EBROOT{module_name.upper()}", "./")).expanduser()

        exec_file = module_root / str(self.metadata.run.exec_file)

        exec_file.resolve()

        try:
            exec_file.relative_to(module_root)
        except ValueError:
            logger.error(f"Try to call external programs [{exec_file}]! module_root={module_root}")
            raise RuntimeError(f"It is forbidden to call external programs! [{exec_file}]!  module_root={module_root}")

        command = []

        if not exec_file.exists():
            raise FileExistsError(exec_file)
        elif exec_file.suffix in SpModuleLocal.script_call.keys():
            command = [SpModuleLocal.script_call[exec_file.suffix], exec_file.as_posix()]
        elif os.access(exec_file, os.X_OK):
            command = [exec_file.as_posix()]
        else:
            raise TypeError(f"File '{exec_file}'  is not executable!")

        cmd_arguments = str(self.metadata.run.arguments)

        try:
            arguments = cmd_arguments.format_map(collections.ChainMap({"VAR_ARGS": args}, kwargs,  self.envs))
        except KeyError as key:
            raise KeyError(f"Missing argument {key} ! [ {cmd_arguments} ]")

        command.extend(shlex.split(arguments))

        working_dir = self.envs.get("WORKING_DIR", "./")

        # @ref: https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
        try:
            # exit_status = subprocess.run(
            #     command,
            #     env=collections.ChainMap(self._envs, os.environ),
            #     capture_output=False,
            #     check=True,
            #     shell=True,
            #     text=True,
            #     cwd=working_dir
            # )
            command_line_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # env=self.envs,
                shell=True,
                cwd=working_dir
            )
            # process_output, _ = command_line_process.communicate()

            with command_line_process.stdout as pipe:
                for line in iter(pipe.readline, b''):  # b'\n'-separated lines
                    logger.info(line)

            exitcode = command_line_process.wait()

        except (OSError, subprocess.CalledProcessError) as error:
            logger.error(
                f"""Command failed! [{command}]
                   STDOUT:[{error.stdout}]
                   STDERR:[{error.stderr}]""")
            raise error

        outputs = {
            "EXITCODE": exitcode,
            "WORKING_DIR": working_dir}

        for p_out in self.metadata.out_ports:
            p_name = str(p_out["name"])
            outputs[p_name] = DataObject(_metadata=p_out,  working_dir=working_dir)

        return AttributeTree(outputs)
