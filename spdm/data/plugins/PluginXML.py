import numpy as np
from xml.etree import (ElementTree, ElementInclude)
from spdm.util.LazyProxy import LazyProxy
from ..Collection import FileCollection
from ..Document import Document
from ..Handler import Handler, RefLinker
import h5py
import numpy
import collections
import pathlib
from typing import (Dict, Any)
from spdm.util.logger import logger


class XMLHolder:
    def __init__(self, files, *args, mode=mode, **kwargs):
        self._trees = self.load_mapping(files)

    @property
    def trees(self):
        return self._trees

    def load_mapping(self, path):
        if isinstance(path, str):
            return self.load_mapping(pathlib.Path(path))
        elif isinstance(path, collections.abc.Sequence):
            trees = []
            for fp in path:
                trees.extend(self.load_mapping(fp))
            return trees
        elif path.is_dir():
            trees = []
            for fp in path.glob("*.XML"):
                trees.extend(self.load_mapping(fp))
            return trees

        root = ElementTree.parse(path).getroot()

        # for child in root.findall("{http://www.w3.org/2001/XInclude}include"):
        #     fp = mapping_file.parent/child.attrib["href"]
        #     try:
        #         root.insert(0, ElementTree.parse(fp).getroot())
        #     except ElementTree.ParseError as error:
        #         raise RuntimeError(f"Parse Error in {fp}: {error}")

        #     root.remove(child)

        logger.debug(f"Loading mapping file from {path}")

        return [root]


class XMLHandler(Handler):
    def __init__(self, next_handler, mapping_files, *args, mapper=None, **kwargs):
        super().__init__(*args, **kwargs)

        def default_mapper(xtree, path):
            xpath = ""
            for p in path:
                if type(p) is int:
                    xpath += f"[{p+1}]"
                elif isinstance(p, str):
                    xpath += f"/{p}"
                else:
                    # TODO: handle slice
                    raise TypeError(f"Illegal path type! {type(p)} {path}")

            if len(xpath) > 0 and xpath[0] == "/":
                xpath = xpath[1:]

            return xtree.find(xpath) if xpath != "" else None

        if mapper is None:
            self._mapper = default_mapper
        else:
            self._mapper = lambda xtree, path: mapper(xtree, path)

    def load_mapping(self, path):
        logger.debug(path)
        if isinstance(path, str):
            return self.load_mapping(pathlib.Path(path))
        elif isinstance(path, collections.abc.Sequence):
            trees = []
            for fp in path:
                trees.extend(self.load_mapping(fp))
            return trees
        elif not isinstance(path, pathlib.Path):
            return []
        elif path.is_dir():
            trees = []
            for fp in path.glob("*.xml"):
                trees.extend(self.load_mapping(fp))
            return trees

        root = ElementTree.parse(path).getroot()

        # for child in root.findall("{http://www.w3.org/2001/XInclude}include"):
        #     fp = mapping_file.parent/child.attrib["href"]
        #     try:
        #         root.insert(0, ElementTree.parse(fp).getroot())
        #     except ElementTree.ParseError as error:
        #         raise RuntimeError(f"Parse Error in {fp}: {error}")

        #     root.remove(child)

        logger.debug(f"Loading mapping file from {path}")

        return [root]

    def put(self, grp, path, value, **kwargs):
        raise NotADirectoryError()

    def get(self, trees, path, **kwargs):
        obj = None

        if isinstance(trees, ElementTree.ElementTree):
            trees = [trees]

        for tree in trees:
            obj = self._mapper(tree, p)
            if obj is not None:
                break

        if not isinstance(obj, ElementTree.Element):
            return obj

        dtype = obj.attrib.get("dtype", None)
        res = None
        if dtype is None:
            res = obj
        elif dtype == "ref":
            return RefLinker(obj.attrib.get("schema", None), obj.text)
        elif dtype == "string":
            res = obj.text.split(',')
        elif dtype == "int":
            res = [int(v) for v in obj.text.split(',')]
        elif dtype == "float":
            res = [float(v) for v in obj.text.split(',')]
        else:
            raise NotImplementedError(f"Not supported dtype {dtype}!")

        dims = [int(v) for v in obj.attrib.get("dims", "").split(',') if v != '']

        res = np.array(res)
        if len(dims) == 0 and len(res) == 1:
            res = res[0]
        elif len(dims) > 0:
            res = np.array(res).reshape(dims)

        return res


def connect_xml(uri, *args, filename_pattern="{_id}.h5", handler=None, **kwargs):

    path = pathlib.Path(getattr(uri, "path", uri))

    Document(
        root=XMLHolder(uri, mode=mode),
        handler=XMLHandler()
    )

    return FileCollection(path, *args,
                          filename_pattern=filename_pattern,
                          document_factory=lambda fpath, mode:,
                          **kwargs)


__SP_EXPORT__ = connect_XML
