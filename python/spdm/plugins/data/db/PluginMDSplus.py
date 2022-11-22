import collections
import os
import pathlib
import re
from functools import cached_property

import MDSplus as mds
import numpy as np
from spdm.data.Collection import Collection
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.util.dict_util import format_string_recursive
from spdm.logger import logger
from spdm.util.urilib import urisplit, uriunsplit


class MDSplusEntry(Entry):

    def __init__(self, holder,  /, **kwargs):
        super().__init__(**kwargs)
        self._holder: MDSplusFile = holder

    def get(self,  *args, **kwargs):
        return self._holder.fetch(*args, **kwargs)

    def put(self,  path, value, *args, **kwargs):
        return self._holder.update({path: value}, *args, **kwargs)

    def iter(self,  path, *args, **kwargs):
        return self._holder.iter(path, *args, **kwargs)


def open_mdstree(tree_name, shot,  mode="NORMAL", path=None):
    if tree_name is None:
        raise ValueError(f"Treename is empty!")
    try:
        shot = int(shot)
        logger.info(f"Open MDSTree: tree_name={tree_name} shot={shot} mode=\"{mode}\" path='{path}'")
        tree = mds.Tree(tree_name, shot, mode=mode, path=path)
    except mds.mdsExceptions.TreeFOPENR as error:
        # tree_path = os.environ.get(f"{tree_name}_path", None)
        raise FileNotFoundError(
            f"Can not open mdsplus tree! tree_name={tree_name} shot={shot} tree_path={path} mode={mode} \n {error}")
    except mds.mdsExceptions.TreeNOPATH as error:
        raise FileNotFoundError(
            f"{tree_name}_path is not defined! tree_name={tree_name} shot={shot}  \n {error}")
    return tree


class MDSplusFile(File):
    MDS_MODE = {
        File.Mode.r: "ReadOnly",
        File.Mode.w: "Normal",
        File.Mode.r | File.Mode.w: "Normal",
        File.Mode.a: "Edit",
        File.Mode.x: "New"
    }

    def __init__(self, *args, tree_name=None,   **kwargs):
        super().__init__(*args,  ** kwargs)

        self._envs = {}
        # {k: (v if not isinstance(v, slice) else f"{v.start}:{v.stop}:{v.step}")
        #   for k, v in self._envs.items()}

        query = self._metadata.get("query", None) or {}

        if tree_name is None:
            tree_name = query.get("tree_name", None)

        self._mds_mode = MDSplusFile.MDS_MODE[self.mode]
        self._tree_name = tree_name
        self._shot = query.get("shot", None)
        if self._shot is None:
            self._shot = self._metadata.get("fragment", 0)
        self._path = self.path
        self._entry = MDSplusEntry(self)

    def __del__(self):
        pass

    def read(self, lazy=True) -> Entry:
        return self._entry

    def write(self, d):
        raise NotImplementedError()

    def fetch(self, request, *args,   **kwargs):
        if not request:
            return self

        if isinstance(request, str):
            request = {"query": request}

        request = collections.ChainMap(request, kwargs)

        # elif isinstance(request, collections.abc.Mapping):

        tree_name = request.get("@tree", self._tree_name)

        tdi = request.get("query", None) or request.get("@text", None)

        # elif isinstance(request, str):
        #     tdi = request
        # else:
        #     raise ValueError(request)

        if not tdi:
            return self

        tdi = tdi.format_map(self._envs)

        mode = self._mds_mode
        shot = self._shot
        path = self.path

        res = None
        try:
            with mds.Tree(tree_name, int(shot), mode=mode, path=path) as tree:
                res = tree.tdiExecute(tdi).data()
        except mds.mdsExceptions.TdiException as error:
            raise RuntimeError(f"MDSplus TDI error [{tdi}]! {error}")
        except mds.mdsExceptions.TreeFOPENR as error:
            raise FileNotFoundError(
                f"Can not open mdsplus tree! tree_name={tree_name} shot={shot} tree_path={path} mode={mode} \n {error}")
        except mds.mdsExceptions.TreeNOPATH as error:
            raise FileNotFoundError(f"{tree_name}_path is not defined! tree_name={tree_name} shot={shot}  \n {error}")
        except mds.mdsExceptions.TreeNODATA as error:
            logger.error(f"No data! tree_name={tree_name} shot={shot} tdi=\"{tdi}\" \n {error}")
        except Exception as error:
            raise error

        if not isinstance(res, np.ndarray):
            pass
        elif len(res.shape) == 2:
            if res.shape[1] == 1:
                res = res[:, 0]
            elif res.shape[0] == 1:
                res = res[0]
            else:
                res = res.transpose(1, 0)
        return res

    def update(self, request, *args,  envs=None, **kwargs):
        raise NotImplementedError()


class MDSplusCollection(Collection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

    def insert_one(self, fid=None, *args,  query=None, mode=None, **kwargs):
        fid = fid or self.guess_id(
            *args, **collections.ChainMap((query or {}), kwargs)) or self.next_id
        return MDSplusFile(self.metadata, fid=fid, mode=mode or "w", **kwargs)

    def find_one(self, fid=None, *args, query=None, projection=None, mode=None, **kwargs):
        fid = fid or self.guess_id(
            *args, **collections.ChainMap((query or {}), kwargs))
        return MDSplusFile(self.metadata, fid=fid, mode=mode or "w", **kwargs).fetch(projection)

    def count(self, predicate=None, *args, **kwargs) -> int:
        return NotImplemented()

    # def find_one(self, predicate: Document,  projection: Document = None, *args, **kwargs):
    #     shot = getitem(predicate, "shot", None) or getitem(predicate, "_id", None)
    #     if shot is not None:
    #         return MDSplusEntry(self._tree_name, shot, mode="r") .fetch(projection)
    #     else:
    #         for shot in self._foreach_shot():
    #             res = MDSplusEntry(self._tree_name, shot, mode="r").fetch_if(
    #                 projection, predicate)
    #             if res is not None:
    #                 return res
    #     return None

    # def _foreach_shot(self):
    #     f_prefix = f"{self._tree_name.lower()}_"
    #     f_prefix_l = len(f_prefix)
    #     glob = f"{f_prefix}*.tree"
    #     for fp in self._path.glob(glob):
    #         yield fp.stem[f_prefix_l:]

    # def find(self, predicate: Document = None, projection: Document = None, *args, **kwargs):

    #     for shot in self._foreach_shot():
    #         res = MDSplusEntry(self._tree_name, shot, mode="r").fetch_if(projection, predicate)
    #         logger.debug(res)

    #         if res is not None:
    #             yield res

    # def insert_one(self, document: Document, *args, **kwargs):
    #     self._count += 1

    #     shot = int(document.get("shot", self._count))

    #     MDSplusEntry(self._tree_name, shot, mode="x").update(document)

    #     return shot


__SP_EXPORT__ = MDSplusFile
