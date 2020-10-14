import numpy as np
from xml.etree import (ElementTree, ElementInclude)
from spdm.util.LazyProxy import LazyProxy
from ..Collection import FileCollection
from ..Document import Document
from ..Handler import Handler, Linker
import h5py
import numpy
import collections
import pathlib
from typing import (Dict, Any)
from spdm.util.logger import logger


class XMLHandler(Handler):
    def __init__(self,  *args, mapper=None, **kwargs):
        super().__init__(*args, **kwargs)

        def default_mapper(path):
            xpath = ""
            for p in path:
                if type(p) is int:
                    xpath += f"[{p+1}]"
                elif isinstance(p, str):
                    xpath += f"/{p}"
                else:
                    # TODO: handle slice
                    raise TypeError(f"Illegal path type! {type(p)} {path}")
                prev = p

            if len(xpath) > 0 and xpath[0] == "/":
                xpath = xpath[1:]

            return xpath, {}

        self._mapper = mapper or default_mapper

    def convert(self, obj, query={}, lazy=True):
        if not isinstance(obj, ElementTree.Element):
            return obj

        dtype = obj.attrib.get("dtype", None)
        res = None

        if len(obj) > 0 and lazy:
            res = LazyProxy(obj, handler=self)
        elif len(obj) > 0:
            d = {child.tag: self.convert(child, True) for child in obj}
            res = collections.namedtuple(obj.tag, d.keys())(**d)
        elif dtype is None:
            res = obj.text
        elif dtype == "NONE":
            res = None
        elif dtype == "ref":
            res = Linker(obj.attrib.get("schema", None), obj.text.format_map(query or {}))
        else:
            if dtype == "string":
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

    def find(self, trees, path):
        if not isinstance(trees, collections.abc.Sequence):
            trees = [trees]
        path, query = self._mapper(path)
        obj = None
        for tree in trees:
            obj = tree.find(path)
            if obj is not None:
                break
        return obj, query

    def put(self, grp, path, value, **kwargs):
        raise NotImplementedError()

    def get(self, trees, path, **kwargs):
        return self.convert(*self.find(trees, path))

    def get_value(self, trees, path, *args, **kwargs):
        return self.convert(*self.find(trees, path), lazy=False)

    def iter(self, trees, path, **kwargs):
        path, query = self._mapper(path)
        for tree in trees:
            for child in tree.iterfind(path):
                obj = self.convert(child, query)
                if obj is not None:
                    yield obj


def load_xml(path):
    if isinstance(path, str):
        return load_xml(pathlib.Path(path))
    elif isinstance(path, collections.abc.Sequence):
        trees = []
        for fp in path:
            trees.extend(load_xml(fp))
        return trees
    elif not isinstance(path, pathlib.Path):
        return []
    elif path.is_dir():
        trees = []
        for fp in path.glob("*.xml"):
            trees.extend(load_xml(fp))
        return trees
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as msg:
        raise RuntimeError(f"ParseError: {path}: {msg}")
    logger.debug(f"Loading XML file from {path}")

    return [root]
    # for child in root.findall("{http://www.w3.org/2001/XInclude}include"):
    #     fp = xml_file.parent/child.attrib["href"]
    #     try:
    #         root.insert(0, ElementTree.parse(fp).getroot())
    #     except ElementTree.ParseError as error:
    #         raise RuntimeError(f"Parse Error in {fp}: {error}")

    #     root.remove(child)


def open_xml(path, mapper=None, **kwargs):
    return Document(root=load_xml(path), handler=XMLHandler(mapper=mapper))


# def connect_xml(uri, *args, filename_pattern="{_id}.h5", handler=None, **kwargs):

#     path = pathlib.Path(getattr(uri, "path", uri))

#     Document(
#         root=XMLHolder(uri, mode=mode),
#         handler=XMLHandler()
#     )

#     return FileCollection(path, *args,
#                           filename_pattern=filename_pattern,
#                           document_factory=lambda fpath, mode:,
#                           **kwargs)


# __SP_EXPORT__ = connect_XML
