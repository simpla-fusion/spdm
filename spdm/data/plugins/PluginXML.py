import numpy as np
from xml.etree import (ElementTree, ElementInclude)
from spdm.util.LazyProxy import LazyProxy
from ..Collection import FileCollection
from ..Document import Document
from ..Handler import (Holder, Handler, Linker)
import h5py
import numpy
import collections
import pathlib
from typing import (Dict, Any)
from spdm.util.logger import logger


def merge_xml(first, second):
    if first is None or second is None or first.tag != second.tag:
        return

    for child in second:
        id = child.attrib.get("id", None)
        if id is not None:
            target = first.find(f"{child.tag}[@id='{id}']")
        else:
            target = first.find(child.tag)
        if target is not None:
            merge_xml(target, child)
        else:
            first.append(child)


def load_xml(path, *args,  mode="r", **kwargs):
    # TODO: add handler non-local request ,like http://a.b.c.d/babalal.xml
    if isinstance(path, str):
        # o = urisplit(uri)
        path = pathlib.Path(path)

    if isinstance(path, collections.abc.Sequence):
        root = load_xml(path[0], mode=mode)
        for fp in path[1:]:
            merge_xml(root, load_xml(fp, mode=mode))
        return root
    elif not isinstance(path, pathlib.Path) or not path.is_file():
        raise FileNotFoundError(path)

    try:
        root = ElementTree.parse(path).getroot()
        logger.debug(f"Loading XML file from {path}")

    except ElementTree.ParseError as msg:
        raise RuntimeError(f"ParseError: {path}: {msg}")

    for child in root.findall("{http://www.w3.org/2001/XInclude}include"):
        fp = path.parent/child.attrib["href"]
        root.insert(0, load_xml(fp))
        root.remove(child)
    return root


class XMLHolder(Holder):
    def __init__(self, obj,  *args,  **kwargs):
        if not isinstance(obj, ElementTree.Element):
            obj = load_xml(obj, *args,  **kwargs)

        super().__init__(obj)


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
            res = LazyProxy(XMLHolder(obj), handler=self)
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

    def find(self, holder, path):
        path, query = self._mapper(path)
        return holder.data.find(path), query

    def put(self, holder, path, value, **kwargs):
        raise NotImplementedError()

    def get(self, holder, path, **kwargs):
        return self.convert(*self.find(holder, path))

    def get_value(self, holder, path, *args, **kwargs):
        return self.convert(*self.find(holder, path), lazy=False)

    def iter(self, holder, path, **kwargs):
        tree = holder.data
        path, query = self._mapper(path)
        for child in tree.iterfind(path):
            obj = self.convert(child, query)
            if obj is not None:
                yield obj


def open_xml(path, mapper=None, **kwargs):
    return Document(root=XMLHolder(path), handler=XMLHandler(mapper=mapper))


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
