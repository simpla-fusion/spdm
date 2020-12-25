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

LMOD_EXEC = "/usr/share/lmod/lmod/libexec/lmod"


class SpModuleLocal(SpObject):
    """Call subprocess/shell command
    {PKG_PREFIX}/bin/xgenray -i {INPUT_FILE} -o {OUTPUT_DIR}
    """
    _description = AttributeTree({
        "in_ports": [{"name": "args", "kind": "VAR_POSITIONAL"},
                     {"name": "kwargs", "kind": "VAR_KEYWORD"}],
        "out_ports": [{"name": "RETURNCODE"},
                      {"name": "STDOUT"},
                      {"name": "STDERR"},
                      {"name": "OUTPUT_DIR"},
                      ],

        "prescript":  [
            "module purge",
            "module load {mod_path}/{version}{tag_suffix}"
        ],

        "run": {
            "exec_file": "${EBROOTGENRAY}/bin/xgenray",
            "arguments": "-i {equilibrium} -c {config} -n {number_of_steps} -o {OUTPUT} ",
        },

        "postscript": "module purge"
    })

    script_call = {
        ".py": sys.executable,
        ".sh": "bash",
        ".csh": "tcsh"
    }

    def __init__(self, *args, parameters=None, only_module_command=True, **kwargs):
        super().__init__()
        self._parameter = parameters or {}
        self._only_module_command = only_module_command

    def __del(self):
        super().__del__()

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
            else:
                raise NotImplementedError(cmd)

        return res

    def preprocess(self):
        super().preprocess()
        self._execute_script(self._description.prescript)

    def postprocess(self):
        self._execute_script(self._description.postscript)
        super().postprocess()

    def execute(self,  *args, envs=None, **kwargs):
        # super().execute(*args, **kwargs)
        module_name = str(self._description.annotation.name)

        module_root = pathlib.Path(os.environ.get(f"EBROOT{module_name.upper()}",)).expanduser()

        exec_file = module_root / str(self._description.run.exec_file)

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

        inputs = {
            "WORKING_DIR": "./"
        }

        cmd_arguments = str(self._description.run.arguments)

        try:
            arguments = cmd_arguments.format_map(collections.ChainMap(inputs, os.environ))
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

        outputs = {}

        return AttributeTree(
            RETURNCODE=exit_status.returncode,
            STDOUT=exit_status.stdout,
            STDERR=exit_status.stderr,
            **outputs
        )
