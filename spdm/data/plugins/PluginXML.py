import collections
import pathlib

import numpy as np


try:
    from lxml.etree import ParseError as _XMLParseError
    from lxml.etree import XPath as _XPath
    from lxml.etree import _Element as _XMLElement
    from lxml.etree import parse as parse_xml
    _HAS_LXML = True
except ImportError:
    from xml.etree.ElementTree import Element as _XMLElement
    from xml.etree.ElementTree import ParseError as _XMLParseError
    from xml.etree.ElementTree import parse as parse_xml
    _XPath = str
    _HAS_LXML = False


from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser

from ..Collection import Collection, FileCollection
from ..Document import Document
from ..Node import Node


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
        root = parse_xml(path.as_posix()).getroot()
        logger.debug(f"Loading XML file from {path}")

    except _XMLParseError as msg:
        raise RuntimeError(f"ParseError: {path}: {msg}")

    for child in root.findall("{http://www.w3.org/2001/XInclude}include"):
        fp = path.parent/child.attrib["href"]
        root.insert(0, load_xml(fp))
        root.remove(child)
    return root


class XMLNode(Node):
    def __init__(self, holder,  *args,    **kwargs):
        super().__init__(*args, **kwargs)
        self._holder = holder

    def xpath(self, path):
        res = "."
        for p in path:
            if type(p) is int:
                res += f"[ @id='{p}' or position()= {p+1} or @id='*']"
            elif isinstance(p, str) and p[0] == '@':
                res += f"[{p}]"
            elif isinstance(p, str):
                res += f"/{p}"
            else:
                # TODO: handle slice
                raise TypeError(f"Illegal path type! {type(p)} {path}")

        if _HAS_LXML:
            res = _XPath(res)
        return res

    def _convert(self, element, path=[], envs={},  lazy=True, projection=None,):
        if not isinstance(element, _XMLElement):
            return element
        res = None

        if len(element) > 0 and lazy:
            res = XMLNode(element, prefix=self._prefix+path, envs={**envs, **self._envs}).entry
        elif "dtype" in element.attrib or (len(element) == 0 and len(element.attrib) == 0):
            dtype = element.attrib.get("dtype", None)

            if dtype == "string" or dtype is None:
                res = [element.text]
            elif dtype == "int":
                res = [int(v) for v in element.text.split(',')]
            elif dtype == "float":
                res = [float(v) for v in element.text.split(',')]
            else:
                raise NotImplementedError(f"Not supported dtype {dtype}!")

            dims = [int(v) for v in element.attrib.get("dims", "").split(',') if v != '']
            if len(dims) == 0 and len(res) == 1:
                res = res[0]
            elif len(dims) > 0 and len(res) != 0:
                res = np.array(res).reshape(dims)
            else:
                res = np.array(res)
        else:
            res = {child.tag: self._convert(child, path=path+[child.tag], lazy=lazy) for child in element}
            for k, v in element.attrib.items():
                res[f"@{k}"] = v

            text = element.text.strip() if element.text is not None else None
            if text is not None and len(text) != 0:
                query = {}
                prev = None
                for p in self._prefix+path:
                    if type(p) is int:
                        query[f"{prev}#id"] = p
                    prev = p
                res["@text"] = text.format(**query, **envs)
        return res

    def put(self,  path, value, *args, **kwargs):
        if not only_one:
            return PathTraverser(path).apply(lambda p,  v=value, s=self, h=holder: s._push(h, p, v))
        else:
            raise NotImplementedError()

    def get(self,  path, *args, only_one=False, **kwargs):
        if not only_one:
            return PathTraverser(path).apply(lambda p: self.get(p, only_one=True, **kwargs))
        else:
            return self._convert(self.xpath(path).evaluate(self._holder), path=path, **kwargs)

    def get_value(self,  path, *args,  only_one=False, **kwargs):
        if not only_one:
            return PathTraverser(path).apply(lambda p: self.get_value(p, only_one=True, **kwargs))
        else:
            obj = self.xpath(path).evaluate(self._holder)
            if isinstance(obj, collections.abc.Sequence) and len(obj) > 0:
                obj = obj[0]
            return self._convert(obj, lazy=False, path=path, **kwargs)

    def iter(self,  path, *args, **kwargs):
        for spath in PathTraverser(path):
            for child in self.xpath(spath).evaluate(self._holder):
                yield self._convert(child, path=spath)


class XMLDocument(Document):
    def __init__(self,  path=[], *args,   **kwargs):
        if isinstance(path, str):
            path = [path]

        if isinstance(path, collections.abc.Sequence):
            root = XMLNode(load_xml(path, *args,  **kwargs))
        elif isinstance(path, Node):
            root = path
        else:
            raise TypeError(path)

        super().__init__(root, *args,  ** kwargs)


class XMLCollection(FileCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError()
