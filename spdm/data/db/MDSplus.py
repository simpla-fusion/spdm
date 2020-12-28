import collections
import os
import pathlib
import re
from functools import cached_property

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.dict_util import DefaultDict
from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

import MDSplus as mds

from ..Collection import Collection
from ..Document import Document
from ..Node import Node


class MDSplusNode(Node):

    def __init__(self, holder, *args, **kwargs):
        super().__init__(holder, *args, **kwargs)

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
        logger.debug(f"Opend MDSTree: tree_name={tree_name} shot={shot} mode=\"{mode}\" path='{path}'")
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

    def __init__(self, desc, *args, tree_name=None, **kwargs):
        super().__init__(desc, *args, ** kwargs)

        mds_mode = MDSplusDocument.MDS_MODE.get(self.description.mode, "NORMAL")

        self._trees = DefaultDict(lambda t, s=self.description.fid, m=mds_mode,
                                  p=str(self.description.path): open_mdstree(t, s, mode=m, path=p))

        if tree_name is None:
            tree_name = str(self.description.tree_name)
        if not tree_name:
            pass
        else:
            self._trees[None] = self._trees[tree_name]

    @property
    def root(self):
        return MDSplusNode(self._trees)


class MDSplusCollection(Collection):
    def __init__(self, desc, *args,  tree_name=None,  **kwargs):
        super().__init__(desc, *args, **kwargs)

        self._default_tree_name = tree_name or self.description.query.tree_name

        # schema = self.description.schema.lower()
        # authority = self.description.authority or ''
        # path = self.description.path or ""
        # # fid = self.description.fragment.shot or None

        # if tree_name is None:
        #     tree_name = self.description.query.tree_name or None

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
        return MDSplusDocument(self.description, fid=fid, mode=mode, tree_name=tree_name or self._default_tree_name)

    def insert_one(self, *args,  query=None, **kwargs):
        oid = self.guess_id(collections.ChainMap((query or {}), kwargs), auto_inc=True)
        doc = self.open_document(oid, mode="w")
        # doc.update(data or kwargs)
        return doc

    def find_one(self, *args, query=None, projection=None,  **kwargs):
        fid = self.guess_id(query or kwargs)
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


__SP_EXPORT__ = MDSplusCollection
