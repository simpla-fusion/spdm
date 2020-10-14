from ..Collection import (Collection, FileCollection)
from ..Document import Document
from ..Handler import (Handler, HandlerProxy)

import collections
import os
import pathlib
import re
import MDSplus as mds
import numpy as np
from spdm.util.urilib import urisplit
from spdm.util.logger import logger


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

class MDSplusHolder(object):
    def __init__(self, tree_name, fid, *args, mode="r", **kwargs):
        self._tree = mds.Tree(tree_name, fid, mode="NORMAL")
        logger.debug(f"Opend MDSTree: {tree_name} {fid} mode=\"{mode}\"")

    @property
    def tree(self):
        return self._tree


class MDSplusHandler(Handler):

    def put(self, holder, path, value, **kwargs):
        raise NotImplementedError()

    def get(self, holder, path=None, projection=None):
        if path is None:
            return None
        elif not isinstance(path, str):
            raise NotImplementedError()
        else:
            return holder.tree.tdiExecute(path).data()


class MDSplusCollection(Collection):
    def __init__(self, uri, *args,  **kwargs):
        super().__init__(*args, handler=MDSplusHandler(), **kwargs)

        if isinstance(uri, str):
            uri = urisplit(uri)

        authority = getattr(uri, "authority", None)

        path = pathlib.Path(getattr(uri, "path", None)).resolve().expanduser()

        self._tree_name = path.stem

        if authority == "":
            os.environ[f"{self._tree_name}_path"] = path.as_posix()
        else:
            os.environ[f"{self._tree_name}_path"] = f"{authority}:{path.as_posix()}"

        logger.debug(f"Open MDSplus collection:[{os.environ[f'{self._tree_name}_path']}]")

    def open_document(self, fid, mode):
        return Document(root=MDSplusHolder(self._tree_name, fid, mode="NORMAL"),
                        handler=self._handler,
                        collection=self)

    def insert_one(self, data=None, *args, **kwargs):
        doc = self.open_document(self.guess_id(data or kwargs, auto_inc=True), mode="w")
        doc.update(data or kwargs)
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


# class MDSplusConnect:


#     def open(self, tree_path, *args, **kwargs):
#         tree_path = tree_path[re.search(r'[^/]', tree_path).start():]

#         if self._netloc is None:
#             return MDSplusRemoteCollection(tree_path, prefix=self._prefix)
#         else:
#             return MDSplusLocalCollection(tree_path, prefix=self._prefix)

__SP_EXPORT__ = MDSplusCollection
