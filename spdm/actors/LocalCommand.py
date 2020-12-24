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


class LocalCommand(SpObject):
    """Call subprocess/shell command
    {PKG_PREFIX}/bin/xgenray -i {INPUT_FILE} -o {OUTPUT_DIR}
    """
    _schema = {
        "in_ports": [{"name": "args", "kind": "VAR_POSITIONAL"},
                     {"name": "kwargs", "kind": "VAR_KEYWORD"}],
        "out_ports": [{"name": "RETURNCODE"},
                      {"name": "STDOUT"},
                      {"name": "STDERR"},
                      {"name": "OUTPUT_DIR"},
                      ],
        "run": {
            "exec_cmd": "",
            "arguments": ""
        }
    }

    script_call = {
        ".py": sys.executable,
        ".sh": "bash",
        ".csh": "tcsh"
    }

    def __init__(self, *args, parameters=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parameter = parameters or {}

    def preprocess(self,  cache, envs):

        exec_cmd = pathlib.Path(self._schema.get("exec_cmd", "").format_map(envs))

        if not exec_cmd.exists():
            raise FileExistsError(exec_cmd)
        elif exec_cmd.suffix in LocalCommand.script_call.keys():
            command = [LocalCommand.script_call[exec_cmd.suffix],
                       exec_cmd.as_posix()]
        elif os.access(exec_cmd, os.X_OK):
            command = [exec_cmd.as_posix()]
        else:
            raise TypeError(f"File '{exec_cmd}'  is not executable!")

        try:
            arguments = self._schema.get("arguments", "").format_map(envs)
        except KeyError as key:
            raise KeyError(
                f"Missing argument {key} in arguments [ {self.__class__.arguments} ]")

        command.extend(shlex.split(arguments))

        return patch

    def run(self,  cache, _envs):
        command = cache.get_r([self.id, "local_vars", "command"], None)
        #  module purge; module load genray/10.8-foss-2019 ; ${EBROOTGENRAY}/bin/xgenray -i ..../. ; module purge
        try:
            exit_status = subprocess.run(
                command,
                env=_envs,
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

        return {"RETURNCODE": exit_status.returncode,
                "STDOUT": exit_status.stdout,
                "STDERR": exit_status.stderr,
                "OUTPUT_DIR": pathlib.Path(_envs.get("WORKING_DIR"))
                }

    def __call__(self, *args, **kwargs):
        return NotImplemented
