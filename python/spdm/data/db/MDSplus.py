import collections
import os
import pathlib
import re
from functools import cached_property

import numpy as np
from spdm.util.dict_util import DefaultDict
from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

import MDSplus as mds

from ..Collection import Collection
from ..Document import Document
from ..Entry import Entry


class MDSplusNode(Entry):

    def __init__(self, holder, *args, **kwargs):
        super().__init__(holder, *args, **kwargs)

    # def __del__(self):
    #     logger.info(f"Close {self.__class__.__name__}")

    def get(self, path, *args, projection=None,  **kwargs):
        if isinstance(path, collections.abc.Mapping):
            tree_name = path.get("@tree_name", None)
            path = path.get("@text", None)
        res = None
        if isinstance(path, str) and len(path) > 0:
            try:
                res = self.holder[tree_name].tdiExecute(path)
            except mds.mdsExceptions.TdiException as error:
                raise RuntimeError(f"MDSplus TDI error [{path}]! {error}")
            # except mds.mdsExceptions.TdiINV_SIZE as error:
            #     raise SyntaxError(f"MDSplus TDI syntax error [{path}]! {error}")
            res = res.data()
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

    def put(self,  path, value, *args, **kwargs):
        raise NotImplementedError()

    def iter(self,  path, *args, **kwargs):
        raise NotImplementedError(path)


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


class MDSplusDocument(Document):
    MDS_MODE = {
        "r": "ReadOnly",
        "rw": "Normal",
        "w": "Normal",
        "w+": "Edit",
        "x": "New"
    }

    def __init__(self, *args, desc=None,  tree_name=None, path=None, **kwargs):
        super().__init__(*args,  path=path, ** kwargs)

        mds_mode = MDSplusDocument.MDS_MODE.get(self.mode, "NORMAL")

        self._trees = DefaultDict(lambda t, s=self.fid, m=mds_mode,
                                  p=str(self.path): open_mdstree(t, s, mode=m, path=p))

        self._trees[None] = self._trees[tree_name or (desc or {}).get("tree_name")]

    # def __del__(self):
    #     # del self._trees
    #     # super().__del__()
    #     pass
    #     # for k, tree in self._trees.items():
    #     #     logger.debug(tree)

    def close(self):
        self._trees.clear()

    @property
    def root(self):
        return MDSplusNode(self._trees)


class MDSplusCollection(Collection):
    def __init__(self, _metadata=None, *args,  tree_name=None,  **kwargs):
        super().__init__(_metadata, *args,  **kwargs)

        if isinstance(_metadata, str):
            o = urisplit(_metadata)
            self._authority = o["authority"]
            self._path = o["path"]
            self._tree_name = o["query"]["tree_name"]

        else:
            raise RuntimeError(_metadata)
        # schema = self.metadata.schema.lower()
        # authority = self.metadata.authority or ''
        # path = self.metadata.path or ""
        # # fid = self.metadata.fragment.shot or None

        # if tree_name is None:
        #     tree_name = self.metadata.query.tree_name or None

        # self._default_tree_name = tree_name

        # if schema != "mdsplus":
        #     raise NotImplementedError(schema)

        # # self._path = uriunsplit(schema, "" if not authority else authority, path)
        # if not authority:
        #     self._path = path
        # else:
        #     self._path = uriunsplit("mdsplus", str(authority), path)

        # self.add_tree(tree_name, uriunsplit(schema, "" if not authority else authority, path))

    # def __del__(self):
    #     for tree_name in self._trees:
    #         del os.environ[f"{tree_name}_path"]

    def open_document(self, fid, mode="r", tree_name=None):
        return MDSplusDocument(None, desc=self.metadata, fid=fid, mode=mode,
                               path=self._path, tree_name=tree_name or self._tree_name)

    def insert_one(self, *args,  query=None, **kwargs):
        fid = self.guess_id(*args, **collections.ChainMap((query or {}), kwargs))
        doc = self.open_document(fid, mode="w")
        # doc.update(data or kwargs)
        return doc

    def find_one(self, *args, query=None, projection=None,  **kwargs):
        fid = self.guess_id(*args, **collections.ChainMap((query or {}), kwargs))
        # fid=self.guess_id(query or kwargs)
        doc = None
        if fid is not None:
            doc = self.open_document(fid, mode="r")
        else:
            raise NotImplementedError()

        if projection is not None:
            raise NotImplementedError()

        return doc

    def count(self, predicate=None, *args, **kwargs) -> int:
        return 0

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


__SP_EXPORT__ = MDSplusDocument
