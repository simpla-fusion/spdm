import collections
import fnmatch
import functools
import inspect
import io
import pathlib
import pprint
from typing import Any, Dict

from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.SpURI import SpURI


class DataEntry(object):
    """Entry of data source : file ,db ..."""

    DEFALUT_FILE_FORMAT = "_yaml"
    file_associations = {
        ".bin": "binary",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".nc": "netcdf",
        ".netcdf": "netcdf",
        ".namelist": "namelist",
        ".nml": "namelist",
        ".namelist": "namelist",
        ".json": "_json",
        ".yaml": "_yaml",
        ".txt": "txt",
        ".csv": "csv",
        ".numpy": "numpy"
    }

    @classmethod
    def create(cls, entry, *args, **kwargs):
        return cls(entry, *args, **kwargs)

    @staticmethod
    def __new__(cls, entry, *args, suffix=None, format=None, **kwargs):
        if cls is not DataEntry:
            return super(DataEntry, cls).__new__(entry, *args, **kwargs)
            
        if isinstance(entry, (str, pathlib.Path)):
            entry = SpURI(entry)

        if format is not None:
            suffix = f".{format}"
        plugin_name = None
        if entry is None:
            file_format = DataEntry.file_associations.get(
                suffix, DataEntry.DEFALUT_FILE_FORMAT)
            plugin_name = f"file.{file_format}"
        elif isinstance(entry, SpURI):
            if entry.schema in ['local', 'file', None]:
                file_format = DataEntry.file_associations.get(
                    suffix or entry.path.suffix, DataEntry.DEFALUT_FILE_FORMAT)
                plugin_name = f"file.{file_format}"
            else:
                plugin_name = entry.schema
        elif isinstance(entry, io.IOBase):
            file_format = DataEntry.file_associations.get(
                suffix, DataEntry.DEFALUT_FILE_FORMAT)
            plugin_name = f"file.{file_format}"
        else:
            raise TypeError(f"Illegal argument type! {type(entry)}")

        plugin_name = f"{__package__}.plugins.{plugin_name}"

        try:
            n_cls = sp_find_module(plugin_name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Can not load 'DataEntry' plugin for {plugin_name}! [{suffix}]")
        if not (inspect.isclass(n_cls) and issubclass(n_cls, DataEntry)):
            raise TypeError(
                f"Can not find 'DataEntry' plugin for {plugin_name} !")
        return object.__new__(n_cls)

    def __init__(self, uri, *args,  cache=None, schema=None,  **kwargs):
        super().__init__()
        self._uri = uri
        self._cache = cache
        self._schema = schema

    def __repr__(self):
        return f"<{self.__class__.__name__} uri='{self._uri}' />"

    def copy(self, path=None):
        raise NotImplementedError()

    @property
    def uri(self):
        return self._uri

    @property
    def schema(self):
        return self._schema

    def join(self, path):
        self._uri.join(path)

    def validate(self, path=None):
        return not self._schema

    def pull_cache(self, path=None):
        # if not super().is_empty:
        #     raise NotImplementedError(whoami(self))
        return

    def push_cache(self, path=None, value=None):
        # if not super().is_empty:
        #     raise NotImplementedError(whoami(self))
        return

    # def _query(self, d, predicate: Dict[str, Any]) -> bool:
    #     raise NotImplementedError(whoami(self))

    # def _update(self, d, predicate: Dict[str, Any]) -> bool:
    #     raise NotImplementedError(whoami(self))
    def flush(self, *args, **kwargs):
        raise NotImplementedError()

    def write(self, contents, *args, **kwargs):
        raise NotImplementedError()

    def read(self, *args, **kwargs):
        raise NotImplementedError()

    def do_check_if(self, path, cond):
        return ContainerEntry.do_update(self, path, cond)

    def do_update(self, path, op):
        return ContainerEntry.do_update(self, path, op)

    def do_fetch(self, path):
        if path is None:
            return self._data
        else:
            return ContainerEntry.do_fetch(self, path.split('.'))

    def do_delete(self, path):
        ContainerEntry.do_delete(self, path)
        return self.push_cache()

    def do_insert(self, path, value):
        return True

    def check_if(self, predicate: Dict[str, Any] = None):
        if predicate is None:
            return True
        return functools.reduce(lambda a, b: a and b,
                                [self.do_check_if(k, cond)
                                 for k, cond in predicate.items()],
                                True)

    def fetch(self, proj: Dict[str, Any] = None):
        self.pull_cache()
        if proj is None:
            return self
        elif isinstance(proj, str):
            return self.do_fetch(proj)
        elif isinstance(proj, collections.abc.Mapping):
            return {p: self.do_fetch(p) for p, v in proj.items() if v > 0}
        elif isinstance(proj, collections.abc.Sequence):
            return self.do_fetch(proj)

    def update(self, d: Dict[str, Any], *args, **kwargs):
        self.pull_cache()
        return [self.do_update(p, op, *args, **kwargs) for p, op in d.items()]

    def fetch_if(self,
                 projection: Dict[str, Any],
                 predicate: Dict[str, Any] = None):
        """If `predicate` is not specified or doc match `predicate` return the
                 `projection` of fields else return None.
        If `projection` is not specified return all fields.
        """
        self.pull_cache()
        return self.fetch(projection) \
            if self.check_if(predicate) else None

    def update_if(self,
                  update: Dict[str, Any],
                  predicate: Dict[str, Any] = None):
        """If `predicate` is not specified or `doc` match `predicate` update
        fields and return True else return False.
        """
        self.pull_cache()
        res = self.update(update) if self.check_if(predicate) else False
        self.push_cache()
        return res

    def delete(self, path, *args, **kwargs):
        self.pull_cache()
        self.do_delete(path, *args, **kwargs)
        self.push_cache()

    def contains(self, path, *args, **kwargs):
        self.pull_cache()
        return ContainerEntry.exists(self, path, *args, **kwargs)

    def dir(self, *args, **kwargs):
        self.pull_cache()
        return ContainerEntry.dir(self, *args, **kwargs)
    # {"*.h5":"hdf5"}
    # file_extents = collections.OrderedDict()

    # def __new__(cls,  *args, **kwargs):
    #     instance = None

    #     if cls is DataEntry:
    #         if len(args) < 0:
    #             instance = super(DataEntry, cls).__new__(cls)

    #         elif type(args[0]) is str:
    #             new_cls = None
    #             spec = urllib.parse.urlparse(args[0])
    #             if spec.scheme != "" and spec.scheme != "file":
    #                 plugin_name = spec.scheme
    #             else:
    #                 for pattern, pluginName in DataEntry.file_extents.items():
    #                     if fnmatch.fnmatch(spec.path, pattern):
    #                         plugin_name = pluginName
    #                         break

    #             new_cls = findModule([__package__, "plugins", plugin_name])

    #             if new_cls is None:
    #                 raise ModuleNotFoundError(
    #                     f"Can not find plugin for {args[0]}")

    #             instance = super(DataEntry, cls).__new__(new_cls)

    #         else:
    #             instance = super(DataEntry, cls).__new__(cls)
    #     else:
    #         instance = super(DataEntry, cls).__new__(cls)
    #         # instance.__init__(**spec._asdict(),**kwargs)
    #     return instance


# TODO: search plugins
