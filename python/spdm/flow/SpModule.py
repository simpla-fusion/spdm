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

from ..data.DataObject import DataObject
from ..util.dict_util import format_string_recursive
from ..util.AttributeTree import AttributeTree
from ..util.logger import logger
from ..util.Signature import Signature
from ..util.SpObject import SpObject


class SpModule(SpObject):

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if cls is not SpModule:
            return object.__new__(cls)
        else:
            return SpObject.__new__(cls, *args, **kwargs)

    def __init__(self, *args, envs=None, metadata=None, **kwargs):
        super().__init__(metadata=metadata)

        self._envs = envs or {}
        self._args = args
        self._kwargs = kwargs
        self._inputs = None
        self._outputs = None

    def __del__(self):
        super().__del__()

    @property
    def envs(self):
        return collections.ChainMap(self._envs, self._metadata)

    def inputs(self):
        if self._inputs is not None:
            return self._inputs
        self._inputs = {}

        working_dir = self.envs.get("INPUT_DIR", "./")

        for p_in in self.metadata.in_ports:
            format_string_recursive(p_in, self.envs)
            p_name = str(p_in["name"])
            if p_name == "VAR_ARGS":
                self._inputs["VAR_ARGS"] = DataObject(self._args,
                                                      _metadata=p_in,
                                                      envs=self.envs,
                                                      working_dir=working_dir)
            else:
                if p_name not in self._kwargs:
                    continue

                data = self._kwargs[p_name]

                if isinstance(data, str):
                    data = format_string_recursive(data, self.envs)
                else:
                    format_string_recursive(data, self.envs)

                data = DataObject.create(data, envs=self.envs, working_dir=working_dir)

                self._inputs[p_name] = DataObject.create(data,
                                                         _metadata=p_in,
                                                         envs=self.envs,
                                                         working_dir=working_dir)
        return self._inputs

    def outputs(self):
        if self._outputs is None:
            self._outputs = AttributeTree(self.run())
        return self._outputs

    def preprocess(self):
        logger.debug(f"Preprocess: {self.__class__.__name__}")

    def postprocess(self):
        logger.debug(f"Postprocess: {self.__class__.__name__}")

    def execute(self):
        logger.debug(f"Execute: {self.__class__.__name__}")
        return None

    def run(self):
        self.preprocess()

        error_msg = None

        # try:
        res = self.execute()
        # except Exception as error:
        #     error_msg = error
        #     logger.error(f"{error}")
        #     res = None

        self.postprocess()

        if error_msg is not None:
            raise error_msg

        return res


class SpModuleLocal(SpModule):
    """Call subprocess/shell command
    {PKG_PREFIX}/bin/xgenray -i {INPUT_FILE} -o {OUTPUT_DIR}
    """

    script_call = {
        ".py": sys.executable,
        ".sh": "bash",
        ".csh": "tcsh"
    }

    def __init__(self, *args, working_dir=None, **kwargs):
        super().__init__(*args, **kwargs)

        if working_dir is not None:
            working_dir = pathlib.Path(working_dir)
        else:
            working_dir = pathlib.Path.cwd()

        count = len(list(working_dir.glob(f"{self.__class__.__name__}_*")))

        working_dir /= f"{self.__class__.__name__}_{count}"

        self._envs["WORKING_DIR"] = working_dir
        self._envs["INPUT_DIR"] = working_dir/"inputs"
        self._envs["OUTPUT_DIR"] = working_dir/"outputs"

        working_dir.mkdir()
        self._envs["INPUT_DIR"].mkdir()
        self._envs["OUTPUT_DIR"].mkdir()

        logger.debug(f"Initialize: {self.__class__.__name__} ")

    def __del__(self):
        logger.debug(f"Finialize: {self.__class__.__name__} ")

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

    def execute(self):
        # super().execute(*args, **kwargs)
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
            envs = collections.ChainMap(self.inputs(), self.envs)
            arguments = cmd_arguments.format_map(envs)
        except KeyError as key:
            raise KeyError(f"Missing argument {key} ! [ {cmd_arguments} ]")

        command.extend(shlex.split(arguments))

        logger.debug(command)

        working_dir = self.envs.get("INPUT_DIR", "./")

        try:
            exit_status = subprocess.run(
                command,
                env=collections.ChainMap(self._envs, os.environ),
                capture_output=True,
                check=True,
                shell=True,
                text=True,
                cwd=working_dir
            )
        except subprocess.CalledProcessError as error:
            logger.error(
                f"""Command failed! [{command}]
                   STDOUT:[{error.stdout}]
                   STDERR:[{error.stderr}]""")
            raise error

        outputs = {"RETURNCODE": exit_status.returncode,
                   "STDOUT": exit_status.stdout,
                   "STDERR": exit_status.stderr, }

        for p_out in self.metadata.out_ports:
            p_name = str(p_out["name"])
            outputs[p_name] = DataObject(p_out,  working_dir=self.envs.get("OUTPUT_DIR", "./"))

        return AttributeTree(outputs)
