import functools
import os
import pprint
import sys

from spdm.data.DataObject import DataObject
from spdm.data.File import File
from spdm.flow.ModuleRepository import ModuleRepository
from spdm.flow.SpModule import SpModule
from spdm.util.LazyProxy import LazyProxy

REPO_NAME = 'FUYUN'

REPO_TAG = 'FY'

os.environ.setdefault(f"{REPO_NAME}_CONFIGURE_PATH",
                      f"pkgdata://{__package__}/../../examples/data/FuYun/configure.yaml")

os.environ.setdefault(f"{REPO_NAME}_OUTPUT_PATH", f"~/{REPO_TAG.lower()}_output")

_module = ModuleRepository(repo_name=REPO_NAME, repo_tag=REPO_TAG)
_module.configure()
_module.factory.insert_handler("SpModule", SpModule)


def __getattr__(path):
    return LazyProxy(_module, [path], handler=lambda o, p: o.new_class_by_path(p))
