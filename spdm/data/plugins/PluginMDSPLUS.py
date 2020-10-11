from ..Collection import (Collection, Document, InsertOneResult, UpdateResult)
from ..Document import Document
import collections
import os
import pathlib
import re
import MDSplus as mds
import numpy as np


# class MDSplusEntry(DataEntry):
#     def __init__(self, tree_name, shot, mode="rw", *args, **kwargs):
#         self._tree_name = tree_name
#         self._shot = shot
#         self._tree = None
#         self._mode = mode

#     def __del__(self):
#         if 'x' in self._mode and self._tree is not None:
#             self._tree.write()
#             self._tree.close()
#             self._tree = None

#     def pull_cache(self):
#         if "x" in self._mode:
#             mds_mode = "NEW"
#         elif "w" in self._mode:
#             mds_mode = "EDIT"
#         elif "r" == self._mode:
#             mds_mode = "READONLY"
#         else:
#             mds_mode = "NORMAL"
#         if self._tree is None:
#             self._tree = mds.Tree(self._tree_name, int(self._shot), mode=mds_mode)
#         return self._tree

#     def push_cache(self):
#         logger.warning("DO NOTHING!")

#     def do_check(self, path, cond):
#         return cond is None

#     def do_fetch(self, path):
#         path = f"{path.replace('.',':').upper()}"
#         logger.debug(path)
#         return self._tree.getNode(path).getData()

#     def _update(self, node, path, value):
#         if isinstance(path, str):
#             path = path.split('.')
#         if len(path) > 0:
#             self._update(node.addNode(path[0]), path[1:], value)
#         elif isinstance(value, str):
#             node.putData(value)
#         elif isinstance(value, collections.abc.Mapping):
#             for k, v in value.items():
#                 self._update(node, k, v)
#         else:
#             node.putData(value)

#     def do_update(self, path, value):
#         return self._update(self._tree, path, value)

#     def do_delete(self, path):
#         raise NotImplementedError(whoami(self))

#     def exists(self, path, *args, **kwargs):
#         self.pull_cache()
#         raise NotImplementedError(whoami(self))

#     def dir(self, *args, **kwargs):
#         self.pull_cache()
#         return self._tree.dir()
#         # raise NotImplementedError(whoami(self))


class MDSplusRemoteCollection(Collection):
    def __init__(self, path, *args, netloc=None, **kwargs):
        raise NotImplementedError(whoami(self))


class MDSPlusCollection(Collection):
    '''
    '''

    def __init__(self, path, *args, session=None, prefix=None, mode="rw",
                 file_mask=0o755, **kwargs):
        self._session = session
        self._mode = mode
        self._count = 0

        if prefix is None or prefix == "":
            prefix = pathlib.Path.cwd()
        elif isinstance(prefix, str):
            prefix = pathlib.Path(prefix)
        elif isinstance(prefix, os.PathLike):
            prefix = prefix
        else:
            raise TypeError(
                f"Only acccept str or os.PathLike not {type(prefix)}")
        self._path = prefix/path
        self._tree_name = self._path.stem
        if 'w' in self._mode:
            self._path.mkdir(mode=file_mask, parents=True, exist_ok=True)

        if not self._path.is_dir():
            raise OSError(f"Can not find path {self._path}")

        os.environ[f"{self._tree_name.lower()}_path"] = str(self._path)

    def __del__(self):
        del os.environ[f"{self._tree_name.lower()}_path"]

    def find_one(self, predicate: Document,  projection: Document = None, *args, **kwargs):
        shot = getitem(predicate, "shot", None) or getitem(predicate, "_id", None)
        if shot is not None:
            return MDSplusEntry(self._tree_name, shot, mode="r") .fetch(projection)
        else:
            for shot in self._foreach_shot():
                res = MDSplusEntry(self._tree_name, shot, mode="r").fetch_if(
                    projection, predicate)
                if res is not None:
                    return res
        return None

    def _foreach_shot(self):
        f_prefix = f"{self._tree_name.lower()}_"
        f_prefix_l = len(f_prefix)
        glob = f"{f_prefix}*.tree"
        for fp in self._path.glob(glob):
            yield fp.stem[f_prefix_l:]

    def find(self, predicate: Document = None, projection: Document = None, *args, **kwargs):

        for shot in self._foreach_shot():
            res = MDSplusEntry(self._tree_name, shot, mode="r").fetch_if(projection, predicate)
            logger.debug(res)

            if res is not None:
                yield res

    def insert_one(self, document: Document,
                   *args, **kwargs) -> InsertOneResult:
        self._count += 1

        shot = int(document.get("shot", self._count))

        MDSplusEntry(self._tree_name, shot, mode="x").update(document)

        return shot

    def replace_one(self, predicate: Document, replacement: Document,
                    *args, **kwargs) -> UpdateResult:
        raise NotImplementedError(whoami(self))

    def update_one(self, predicate: Document, update: Document,
                   *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def delete_one(self, predicate: Document,
                   *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def delete_many(self, predicate: Document,
                    *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def count(self, predicate: Document = None,
              *args, **kwargs) -> int:
        raise NotImplementedError(whoami(self))


class MDSplusConnect:
    __plugin_spec__ = {
        "name": "mdsplus"
    }

    def __init__(self, netloc=None, prefix=None, *args, **kwargs):
        self._netloc = netloc
        self._prefix = prefix

    @classmethod
    def connect(cls, url, *args, **kwargs):
        return MDSplusConnect(url, *args, **kwargs)

    def open(self, tree_path, *args, **kwargs):
        tree_path = tree_path[re.search(r'[^/]', tree_path).start():]

        if self._netloc is None:
            return MDSplusRemoteCollection(tree_path, prefix=self._prefix)
        else:
            return MDSplusLocalCollection(tree_path, prefix=self._prefix)


def connect_mdsplus(uri, *args, filename_pattern="{_id}.h5", handler=None, **kwargs):

    path = pathlib.Path(getattr(uri, "path", uri))

    return Collection(path, *args,
                      filename_pattern=filename_pattern,
                      document_factory=lambda fpath, mode: Document(
                          root=h5py.File(fpath, mode=mode),
                          handler=handler or MDSplusHandler()
                      ),
                      **kwargs)


__SP_EXPORT__ = connect_mdsplus
