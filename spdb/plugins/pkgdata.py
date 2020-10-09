import pathlib
import pkgutil
import uuid
import sys
from spdm.data.DataEntry import DataEntry
from spdm.util.SpURI import SpURI, urisplit
from spdm.util.utilities import merge_dict


class PkgDataEntry(DataEntry):
    """ Default entry for file-like object
    """

    def __init__(self, uri, *args, ** kwargs):
        super().__init__(uri, *args, ** kwargs)
        o = urisplit(uri)
        self._package = o.authority or __package__
        self._path = o.path[1:]
        self._fragment = o.fragment

    def copy(self, path=None):
        return PkgDataEntry(self._uri)

    def _read_all(self):
        pkg = sys.modules.get(self._package, None)

        if hasattr(pkg, "__path__"):  # check namespace package
            for p in pkg.__path__:
                yield DataEntry(pathlib.Path(p)/self._path).read()
        elif hasattr(pkg, "__file__"):  # check normal package
            yield pkgutil.get_data(pkg, self._path)
        else:
            raise ModuleNotFoundError(
                f"Module '{self._package}' is not loaded!")

    def read(self, *args, **kwargs):

        res = {}
        for d in self._read_all():
            merge_dict(res, d)

        return res


__SP_EXPORT__ = PkgDataEntry
