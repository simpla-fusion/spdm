import functools
import os
import pprint
import sys

from spdm.data.DataObject import DataObject
from spdm.data.File import File
from spdm.flow.ModuleRepository import ModuleRepository
from spdm.flow.SpModule import SpModule
from spdm.util.LazyProxy import LazyProxy

os.environ["FUYUN_CONFIGURE_PATH"] = "/home/salmon/workspace/SpDev/SpDB/examples/data/FuYun/configure.yaml"

_module = ModuleRepository(repo_name='FuYun', repo_tag='FY')
_module.configure()
_module.factory.insert_handler("SpModule", SpModule)


def __getattr__(path):
    return LazyProxy(_module, [path], handler=lambda o, p: o.new_class_by_path(p))
