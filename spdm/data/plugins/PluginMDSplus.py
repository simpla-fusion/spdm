import collections
import os
import pathlib
import re

import MDSplus as mds
import numpy as np
from spdm.util.logger import logger
from spdm.util.urilib import urisplit

from ..Collection import Collection, FileCollection
from ..Document import Document
from ..Node import Node


class MDSplusNode(Node):

    def get(self, path, *args, projection=None,  **kwargs):
        if isinstance(path, collections.abc.Mapping):
            path = path.get("@text", None)
        res = None
        if isinstance(path, str) and len(path) > 0:
            try:
                res = self.holder.tdiExecute(path)
            except mds.mdsExceptions.TdiSYNTAX as error:
                raise SyntaxError(f"MDSplus TDI syntax error [{path}]! {error}")
            res = res.data()
        return res

    def put(self,  path, value, *args, **kwargs):
        raise NotImplementedError()

    def iter(self,  path, *args, **kwargs):
        raise NotImplementedError(path)


class MDSplusDocument(Document):
    def __init__(self, tree_name, shot, *args, mode="r", **kwargs):
        try:
            shot = int(shot)
            logger.debug(f"Opend MDSTree: {tree_name} {shot} mode=\"{mode}\"")
            tree = mds.Tree(tree_name, shot, mode="NORMAL")
        except mds.mdsExceptions.TreeFOPENR as error:
            tree_path = os.environ[f"{tree_name}_path"]
            raise FileNotFoundError(
                f"Can not open mdsplus tree! tree_name={tree_name} shot={shot} tree_path={tree_path} \n {error}")

        super().__init__(tree_name, *args, root=MDSplusNode(tree), ** kwargs)


class MDSplusCollection(Collection):
    def __init__(self, uri, *args, tree_name=None,  **kwargs):

        if isinstance(uri, str):
            uri = urisplit(uri)

        authority = getattr(uri, "authority", None)

        path = pathlib.Path(getattr(uri, "path", None)).resolve().expanduser()

        self._tree_name = tree_name or getattr(uri, "fragment", "unnamed")

        if authority == "":
            os.environ[f"{self._tree_name}_path"] = path.as_posix()
        else:
            os.environ[f"{self._tree_name}_path"] = f"{authority}:{path.as_posix()}"

        super().__init__(uri, *args, **kwargs)

    def open_document(self, fid, mode):
        return MDSplusDocument(self._tree_name, fid, mode="NORMAL",  handler=self._handler)

    def insert_one(self, data=None, *args, **kwargs):
        doc = self.open_document(self.guess_id(data or kwargs, auto_inc=True), mode="w")
        # doc.update(data or kwargs)
        return doc

    def find_one(self, predicate=None, projection=None, *args, **kwargs):
        fid = self.guess_id(predicate or kwargs)

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
