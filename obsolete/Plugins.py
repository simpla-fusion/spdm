
import collections
import fnmatch
import functools
import importlib
import pkgutil
import re
import inspect

from .logger import logger
from .Multimap import Multimap

SP_KEYWORD_EXPORT = "__SP_EXPORT__"


class Plugins(Multimap):
    @classmethod
    def get_plugin_name(cls, m):
        return getattr(m, "__plugin_spec__", {"name": m.__name__})["name"]

    def __init__(self, ns_path=None, *args,
                 failsafe="failsafe",
                 export_keyword=None,
                 **kwargs):
        super().__init__(key=lambda m: self.get_plugin_name(m))
        self._export_keyword = export_keyword or SP_KEYWORD_EXPORT
        if ns_path:
            self.regisiter_namespace(ns_path)

        if isinstance(failsafe, str):
            self._failsafe = self.find(failsafe)
        else:
            self._failsafe = failsafe

    def search(self, name: str):
        res = self.find(name)
        if res is None and self._failsafe is not None:
            logger.warning(
                f"Can not find plugin '{name}', failsafe '{self._failsafe.__name__}' is loaded.")
            return self._failsafe
        if res is None:
            raise ModuleNotFoundError(f"Can not find plugin for {name}!")
        return res

    @property
    def failsafe(self):
        return self._failsafe

    def get(self, name, default_value=None):
        return self.find(name) or default_value

    def insert(self, mod):
        if mod is not None:
            mod = getattr(mod, self._export_keyword, mod)

            logger.debug(
                f"Register plugin:  {self.get_plugin_name(mod)} {mod}")
            # if isinstance(self._export_keyword, str):
            return super().insert(mod)

    def regisiter_namespace(self, ns_path):
        ns = importlib.import_module(ns_path)

        # TODO(salmon 2019.7.3) support namespace
        #    Reason: pkgutil.walk_packages does not support PEP420
        #      native namespace packages

        for finder, module_name, ispkg in pkgutil.walk_packages(ns.__path__):
            mod = finder.find_module(module_name).load_module(module_name)
            if ispkg and not hasattr(mod, "__file__"):
                logger.debug(
                    f"Register namespace:  {self.get_plugin_name(mod)} {mod}")
            else:
                self.insert(mod)


class PluginEntry(Entry):
    def __init__(self, prefix=None, export_keyword=None, *args, **kwargs):
        self._prefix = prefix or __package__
        self._export_keyword = export_keyword or SP_KEYWORD_EXPORT

    def canonical_path(self, *path):
        n_path = []
        for p in path:
            if type(p) is str:
                # TODO(salmon) parse slice and in uri template
                n_path.extend(p.split("."))
            elif isinstance(p, collections.abc.Sequence):
                n_path.extend(list(p))
            else:
                n_path.append(p)

        path = ".".join([p for p in n_path if (p != "" and p != None)])
        return f"{self._prefix}.{path}"

    @functools.lru_cache(maxsize=30)
    def fetch(self, path):
        path = self.canonical_path(path)
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            # logger.warning(f"Loading module \"{path}\" failed!")
            raise ModuleNotFoundError(f"Can not load plugin {path}")

        if spec is None:
            return None

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        logger.debug(f"Import module {path}  from [{spec.origin}].")

        return getattr(module, self._export_keyword, None)  \
            or getattr(module, path[path.rfind('.')+1:], None) \
            or module

# __path__ = __import__('pkgutil').extend_path(__path__, __name__)


# import collections
# import fnmatch
# import re
# import pkgutil
# import importlib

# def iter_namespace(ns_pkg):
#     # Specifying the second argument (prefix) to iter_modules makes the
#     # returned name an absolute name instead of a relative one. This allows
#     # import_module to work without having to do additional modification to
#     # the name.
#     return pkgutil.iter_modules(ns_pkg.__path__)


# ns = importlib.import_module(f"{__package__}.plugins")

# _plugins = {
#     name: importlib.import_module(f"{__package__}.plugins.{name}")
#     for finder, name, ispkg in iter_namespace(ns)
# }


# def find_plugin(fp, file_type=None):
#     res = None
#     mod = _plugins.get(file_type, None)
#     if mod is not None:
#         return mod
#     for m in _plugins.values():
#         for partten in m.filename_pattern:
#             if fnmatch.filter(partten, fp) or re.match(fnmatch.translate(partten), fp):
#                 return m

#     raise ValueError(f"Unkown file type {fp} {file_type}!")
