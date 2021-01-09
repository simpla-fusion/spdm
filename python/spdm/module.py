import functools
import os
import pprint
import sys


from spdm.flow.ModuleRepository import ModuleRepository

_repo = None


def _configure(path=None, repo_name=None, repo_tag=None):
    global _repo
    repo_name=repo_name or __package__.split('.')[0]
    if _repo is None:
        _repo = ModuleRepository(repo_name=repo_name, repo_tag=repo_tag)
        # os.environ.setdefault(f"{repo_name.upper()}_CONFIGURE_PATH",
        #                       f"pkgdata://{__package__}/../../examples/data/FuYun/configure.yaml")
        os.environ.setdefault(f"{repo_name.upper()}_OUTPUT_PATH", f"~/{repo_name.lower()}_output")

    _repo.configure(path)


def _get_repo():
    if _repo is None:
        _configure()
    return _repo


def __getattr__(path):
    return _get_repo().entry[path]
    # return LazyProxy(_module, [path], handler=lambda o, p: o.new_class_by_path(p))
