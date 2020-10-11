
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import pathlib
from xml.etree import (ElementTree, ElementInclude)
import numpy as np


class Handler(LazyProxy.Handler):
    DELIMITER = LazyProxy.DELIMITER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HandlerProxy(Handler):
    def __init__(self, next_handler, mapping_file, *args, mapper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_handler = next_handler

        mapping_file = pathlib.Path(mapping_file)

        self._root = ElementTree.parse(mapping_file).getroot()

        for child in self._root.findall("{http://www.w3.org/2001/XInclude}include"):
            fp = mapping_file.parent/child.attrib["href"]
            try:
                self._root.insert(0, ElementTree.parse(fp).getroot())
            except ElementTree.ParseError as error:
                raise RuntimeError(f"Parse Error in {fp}: {error}")

            self._root.remove(child)

        logger.debug(f"Loading mapping from {mapping_file}")

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

            if len(xpath) > 0 and xpath[0] == "/":
                xpath = xpath[1:]

            return self._root.find(xpath) if xpath != "" else None

        if mapper is None:
            self._mapper = default_mapper
        else:
            self._mapper = lambda p: mapper(self._root, p)

    def mapping(self, p):
        obj = self._mapper(p)
        if obj is None or isinstance(obj, str):
            return obj, True

        dtype = obj.attrib.get("dtype", "string")
        if dtype == "ref":
            return obj.text, True
        elif len(obj.getchildren()) > 0:
            return obj, False

        res = None
        if dtype == "string":
            res = obj.text.split(',')
        elif dtype == "int":
            res = [int(v) for v in obj.text.split(',')]
        elif dtype == "float":
            res = [float(v) for v in obj.text.split(',')]
        else:
            raise NotImplementedError(f"Not supported dtype {dtype}!")

        dims = [int(v) for v in obj.attrib.get("dims", "").split(',') if v != '']

        if len(dims) == 0 and len(res) == 1:
            res = res[0]
        elif len(dims) > 0:
            res = np.array(res).reshape(dims)
        else:
            res = np.array(res)

        return res, False

    def put(self, grp, path, value, **kwargs):
        target, is_ref = self.mapping(path)

        if is_ref and target is None:
            self._next_handler.put(grp, path, value, **kwargs)
        elif is_ref and isinstance(target, str):
            self._next_handler.put(grp, target, value, **kwargs)
        else:
            logger.warning(f"Fixed value is not changeable: {path}")

    def get(self, grp, path=[], **kwargs):
        target, is_ref = self.mapping(path)

        if is_ref and target is None:
            return self._next_handler.get(grp, path,  **kwargs)
        elif is_ref and isinstance(target, str):
            return self._next_handler.get(grp, target, **kwargs)
        else:
            return target
