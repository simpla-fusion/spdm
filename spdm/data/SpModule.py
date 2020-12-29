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

from ..util.logger import logger
from ..util.Signature import Signature
from ..util.SpObject import SpObject
from ..util.AttributeTree import AttributeTree
from ..data.DataObject import DataObject

LMOD_EXEC = "/usr/share/lmod/lmod/libexec/lmod"


class SpModule(SpObject):
    def __init__(self, *args, **kwargs):
        super().__init__(name=self.__class__.__name__,
                         label=self.metadata.annotation.label,
                         attributes=kwargs)

        self._output = None
        self._input = AttributeTree()

        # self._parameter = parameters or {}
        # # self._only_module_command = only_module_command
        # self._working_dir = pathlib.Path(f"./{self.__class__.__name__}")

    def __del__(self):
        super().__del__()

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        if self._output is None:
            self._output = AttributeTree(self.run())
        return self._output

    def preprocess(self):
        logger.debug(f"Preprocess: {self.__class__.__name__}")

    def postprocess(self):
        logger.debug(f"Postprocess: {self.__class__.__name__}")

    def execute(self, *args, **kwargs):
        logger.debug(f"Execute: {self.__class__.__name__}")
        return None

    def run(self):
        self.preprocess()

        error_msg = None

        try:
            res = self.execute()
        except Exception as error:
            error_msg = error
            logger.error(f"{error}")
            res = None

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
            self._working_dir = pathlib.Path(working_dir)
        else:
            self._working_dir = pathlib.Path.cwd()

        self._working_dir /= f"{self.__class__.__name__}_{self.uuid}"

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def input_dir(self):
        return self.working_dir/"inputs"

    @property
    def output_dir(self):
        return self.working_dir/"outputs"

    def _execute_module_command(self, *args):
        logger.debug(f"MODULE CMD: module {' '.join(args)}")
        py_commands = os.popen(f"{LMOD_EXEC} python {' '.join(args)}  ").read()
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

    def execute(self,  *args, envs=None, **kwargs):
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

        command = [exec_file]

        if not exec_file.exists():
            raise FileExistsError(exec_file)
        elif exec_file.suffix in SpModuleLocal.script_call.keys():
            command = [SpModuleLocal.script_call[exec_file.suffix], exec_file.as_posix()]
        elif os.access(exec_file, os.X_OK):
            command = [exec_file.as_posix()]
        else:
            raise TypeError(f"File '{exec_file}'  is not executable!")

        mod_envs = {
            "WORKING_DIR": self.working_dir,
            "INPUT_DIR": self.input_dir,
            "OUTPUT_DIR": self.output_dir
        }

        inputs = {}

        for p_in in self.metadata.in_ports:
            
            p_name = str(p_in["name"])
            if p_name == "VAR_ARGS":
                inputs[p_name] = DataObject(p_in, args, working_dir=self.input_dir)
            else:
                inputs[p_name] = DataObject(p_in, kwargs.get(p_name, None), working_dir=self.input_dir)

        cmd_arguments = str(self.metadata.run.arguments)

        try:
            arguments = cmd_arguments.format_map(collections.ChainMap(inputs, mod_envs, os.environ))
        except KeyError as key:
            raise KeyError(f"Missing argument {key} ! [ {cmd_arguments} ]")

        command.extend(shlex.split(arguments))

        logger.debug(command)

        try:
            exit_status = subprocess.run(
                command,
                env=envs,
                capture_output=True,
                check=True,
                shell=False,
                text=True
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
            outputs[p_name] = DataObject(p_out,  working_dir=self.output_dir)

        return AttributeTree(outputs)
