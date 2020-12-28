import collections
import os
import pathlib
import re

import MDSplus as mds
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.dict_util import DefaultDict
from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

from ..Collection import Collection
from .LocalFile import FileCollection
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


def open_mdstree(tree_name, shot,  mode="NORMAL"):
    if tree_name is None:
        raise ValueError(f"Treename is empty!")
    try:
        shot = int(shot)
        logger.debug(f"Opend MDSTree: tree_name={tree_name} shot={shot} mode=\"{mode}\"")
        tree = mds.Tree(tree_name, shot, mode)
    except mds.mdsExceptions.TreeFOPENR as error:
        tree_path = os.environ.get(f"{tree_name}_path", None)
        raise FileNotFoundError(
            f"Can not open mdsplus tree! tree_name={tree_name} shot={shot} tree_path={tree_path} \n {error}")
    except mds.mdsExceptions.TreeNOPATH as error:
        raise FileNotFoundError(
            f"{tree_name}_path is not defined! tree_name={tree_name} shot={shot}   \n {error}")
    return tree


class MDSplusDocument(Document):
    MDS_MODE = {
        "r": "READONLY",
        "rw": "NORMAL",
        "w": "NORMAL",
        "w+": "EDIT",
        "x": "NEW"
    }

    def __init__(self, desc=None, *args, fid=None,  mode="r", tree_name=None,  **kwargs):
        mds_mode = MDSplusDocument.MDS_MODE.get(mode, "NORMAL")

        self._trees = DefaultDict(lambda t, s=fid, m=mds_mode: open_mdstree(t, s, mode=m))

        if tree_name is not None:
            self._trees[None] = self._trees[tree_name]

        super().__init__(desc, *args, fid=fid,
                         root=MDSplusNode(self._trees, tree_name=tree_name),
                         mode=mode, ** kwargs)


class MDSplusCollection(Collection):
    def __init__(self, desc, *args,  tree_name=None,  **kwargs):
        super().__init__(desc, *args, **kwargs)

        schema = (self.envs.schema or "file").split('+')
        authority = self.envs.authority or ''
        path = self.envs.path or ""
        fid = self.envs.fragment.shot or None

        if tree_name is None:
            tree_name = desc.query.tree_name or None

        self._default_tree_name = tree_name

        if len(schema) == 0:
            schema = None
        elif len(schema) == 1:
            schema = schema[0] if schema[0] != "mdsplus" else None
        elif len(schema) == 2:
            assert(schema[0].lower() == "mdsplus")
            schema = schema[1]
        else:
            raise NotImplementedError(schema)

        self.add_tree(tree_name, uriunsplit(schema, authority, path))

    # def __del__(self):
    #     for tree_name in self._trees:
    #         del os.environ[f"{tree_name}_path"]

    def add_tree(self, tree_name, tree_path):
        if (tree_name is None or not tree_path.endswith(tree_name)) and "~t" not in tree_path:
            tree_path = f"{tree_path}/~t/"
        if tree_name is not None:
            os.environ[f"{tree_name}_path"] = tree_path
        else:
            os.environ["default_tree_path"] = tree_path

    def open_document(self, fid, mode, tree_name=None):
        return MDSplusDocument(fid=fid, mode=mode, tree_name=tree_name or self._default_tree_name)

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
