import collections
import pathlib
import typing

import numpy as np
from lxml.etree import Comment as _XMLComment
from lxml.etree import ParseError as _XMLParseError
from lxml.etree import XPath as _XPath
from lxml.etree import _Element as _XMLElement
from lxml.etree import fromstring
from lxml.etree import parse as parse_xml
from spdm.utils.PathTraverser import PathTraverser
from spdm.utils.tags import _not_found_, _undefined_
from spdm.data.Entry import Entry
from spdm.data.File import File
from spdm.utils.tree_utils import format_string_recursive
from spdm.utils.misc import normalize_path, serialize
from spdm.utils.logger import logger
_TPath = typing.TypeVar("_TPath")


def merge_xml(first, second):
    if first is None:
        raise ValueError(f"Try merge to None Tree!")
    elif second is None:
        return first
    elif first.tag != second.tag:
        raise ValueError(
            f"Try to merge tree to different tag! {first.tag}<={second.tag}")

    for child in second:
        if child.tag is _XMLComment:
            continue
        eid = child.attrib.get("id", None)
        if eid is not None:
            target = first.find(f"{child.tag}[@id='{eid}']")
        else:
            target = first.find(child.tag)
        if target is not None:
            merge_xml(target, child)
        else:
            first.append(child)


def load_xml(path, *args,  mode="r", **kwargs):
    # TODO: add handler non-local request ,like http://a.b.c.d/babalal.xml

    if type(path) is list:
        root = None
        for fp in path:
            if root is None:
                root = load_xml(fp, mode=mode)
            else:
                merge_xml(root, load_xml(fp, mode=mode))
        return root
    elif isinstance(path, str):
        path = pathlib.Path(path)

    root = None
    if path.exists() and path.is_file():
        try:
            if isinstance(path, pathlib.Path):
                root = parse_xml(path.as_posix()).getroot()
            elif isinstance(path, str):
                root = parse_xml(path).getroot()
            else:
                raise TypeError(f"Invalid path type: {type(path)}")
            # logger.debug(f"Loading XML file from {path}")
        except _XMLParseError as msg:
            raise RuntimeError(f"ParseError: {path}: {msg}")
    else:
        raise FileNotFoundError(path)

    if root is not None:
        for child in root.findall("{http://www.w3.org/2001/XInclude}include"):
            fp = path.parent/child.attrib["href"]
            root.insert(0, load_xml(fp))
            root.remove(child)

    return root


@Entry.register("xml")
class XMLEntry(Entry):
    def __init__(self, data, *args, envs={}, **kwargs):
        super().__init__({}, *args, **kwargs)

        if not isinstance(data, str):
            pass
        elif not data.strip(" ").startswith("<"):
            data = load_xml(data)
        else:
            data = fromstring(data)
        self._data = data
        self._envs = envs

    def __repr__(self) -> str:
        # return f"<{self.__class__.__name__} root={self._root} path={self._path} />"
        return f"<{self._data.tag}  path=\"{self._path}\" />"

    def __copy__(self) -> Entry:
        other = super().__copy__()
        other._envs = self._envs
        other._data = self._data
        return other

    def _xpath(self, path):
        envs = {}
        res = "."
        prev = None
        for p in path:
            if type(p) is int:
                res += f"[(position()= {p+1} and @id ) or (@id={p+1}) or @id='*']"
                envs[prev] = p
            # # elif isinstance(p, slice):
            # #     if p == slice(None):
            # #         res += "[@id]"
            # #     else:
            # #         raise NotImplementedError("XML DO NOT SUPPORT SLICE!")
            # elif isinstance(p, (tuple, set)):
            #     raise NotImplementedError(f"XML DO NOT SUPPORT TUPLE OR SET!{path}")
            elif isinstance(p, str) and len(p) > 0:
                if p[0] == '@':
                    res += f"[{p}]"
                else:
                    res += f"/{p}"
                    prev = p
            else:
                envs[prev] = p
                # # TODO: handle slice
                # raise TypeError(f"Illegal path type! {type(p)} {path}")

        return res, envs

    def xpath(self, path):
        p, e = self._xpath(path)
        return _XPath(p), e

    def _convert(self, element: _XMLElement, path=[], lazy=True, envs=None, only_one=False, **kwargs):
        if not isinstance(element, list):
            pass
        elif len(element) == 0:
            return None
        elif not isinstance(path[-1], slice) and ((len(element) == 1) or only_one):
            return self._convert(element[0], path=path, lazy=lazy, envs=envs, **kwargs)
        else:
            return [self._convert(e, path=path, lazy=lazy, envs=envs, **kwargs) for e in element]

        res = None
        text = element.text.strip() if element.text is not None else None
        if text is not None and len(text) > 0:
            if "dtype" in element.attrib or (len(element) == 0 and len(element.attrib) == 0):
                dtype = element.attrib.get("dtype", None)
                if dtype == "string" or dtype is None:
                    res = [text]
                elif dtype == "int":
                    res = [int(v.strip())
                           for v in text.strip(',').split(',')]
                elif dtype == "float":
                    res = [float(v.strip())
                           for v in text.strip(',').split(',')]
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
                res = text
        elif not lazy:
            res = {}
            for child in element:
                if child.tag is _XMLComment:
                    continue
                obj = self._convert(
                    child, path=path+[child.tag], envs=envs, lazy=lazy, **kwargs)
                tmp = res.setdefault(child.tag, obj)
                if tmp is obj:
                    continue
                elif isinstance(tmp, list):
                    tmp.append(obj)
                else:
                    res[child.tag] = [tmp, obj]

            # res = {child.tag: self._convert(child, path=path+[child.tag], envs=envs, lazy=lazy, **kwargs)
            #        for child in element if child.tag is not _XMLComment}
            for k, v in element.attrib.items():
                res[f"@{k}"] = v

            text = element.text.strip() if element.text is not None else None
            if text is not None and len(text) != 0:
                query = {}
                prev = None
                for p in self._prefix+path:
                    if type(p) is int:
                        query[f"{prev}"] = p
                    prev = p

                # if not self._envs.fragment:
                #     fstr = query
                # else:
                #     fstr = collections.ChainMap(query, self.envs.fragment.__data__, self.envs.query.__data__ or {})
                # format_string_recursive(text, fstr)  # text.format_map(fstr)
                res["@text"] = text
        else:
            res = XMLEntry(element, prefix=[], envs=envs)

        if envs is not None and isinstance(res, (str, collections.abc.Mapping)):
            res = format_string_recursive(res, collections.ChainMap(envs, self._envs))
        return res

    def insert(self,  *args, **kwargs): return super().insert(*args, **kwargs)

    def update(self,  *args, **kwargs): return super().update(*args, **kwargs)

    def remove(self,  *args, **kwargs): return super().remove(*args, **kwargs)

    def query(self, *args, default_value: typing.Any = _undefined_,  lazy=_undefined_, **kwargs):

        res = super().query(default_value=_not_found_)

        if res is not _not_found_:
            return res

        path = self._path[:]
        xp, envs = self.xpath(path)

        obj = xp.evaluate(self._data)

        if lazy is _undefined_:
            lazy = default_value is _undefined_

        res = self._convert(obj, path=path, lazy=lazy, envs=envs, **kwargs)

        if res is not _not_found_:
            pass
        elif default_value is not _undefined_:
            res = default_value
        else:
            raise RuntimeError(path)

        return res

    def _find(self,  path, *args, only_one=False, default_value=None, projection=None, **kwargs):
        if not only_one:
            res = PathTraverser(path).apply(lambda p: self.find(
                p, only_one=True, default_value=_not_found_, projection=projection))
        else:
            path = self._prefix+normalize_path(path)
            xp, envs = self.xpath(path)
            res = self._convert(xp.evaluate(self._data), lazy=True, path=path, envs=envs, projection=projection)

        if res is _not_found_:
            res = default_value
        return res

    def _get_value(self,  path: typing.Optional[_TPath] = None, *args,  only_one=False, default_value=_not_found_, **kwargs):

        if not only_one:
            return PathTraverser(path).apply(lambda p: self._get_value(p, only_one=True, **kwargs))
        else:
            path = self._prefix+normalize_path(path)
            xp, envs = self.xpath(path)
            obj = xp.evaluate(self._data)
            if isinstance(obj, collections.abc.Sequence) and len(obj) == 1:
                obj = obj[0]
            return self._convert(obj, lazy=False, path=path, envs=envs, **kwargs)

    def find(self,  *args, envs={}, **kwargs):
        # path, s_envs = self._xpath(self._path[:])
        # TODO: PathTraverser(path):
        for s_path in self._path.traversal():
            s_path, s_envs = self._xpath(s_path)
            res = _XPath(s_path).evaluate(self._data)

            if len(res) == 0:
                break
            for child in res:
                if child.tag is not _XMLComment:
                    yield self._convert(child, path=s_path,
                                        envs=collections.ChainMap(s_envs, envs))

    def items(self,    *args, envs={}, **kwargs):
        path = self._path
        for spath in PathTraverser(path):
            xp, s_envs = self.xpath(spath)
            for child in xp.evaluate(self._data):
                if child.tag is _XMLComment:
                    continue
                res = self._convert(child, path=spath,
                                    envs=collections.ChainMap(s_envs, envs))
                yield child.tag, res

    def values(self,    *args, envs={}, **kwargs):
        path = self._path
        for spath in PathTraverser(path):
            xp, s_envs = self.xpath(spath)
            for child in xp.evaluate(self._data):
                if child.tag is _XMLComment:
                    continue
                res = self._convert(child, path=spath,
                                    envs=collections.ChainMap(s_envs, envs))
                yield res

    @ property
    def attribute(self):
        return self._data.attrib

    def __serialize__(self):
        return serialize(self.query(default_value=_not_found_))


class XMLFile(File):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ** kwargs)
        self._root = load_xml(self.uri.path, mode=self.mode)

    def read(self, lazy=True) -> Entry:
        return XMLEntry(self._root, writable=False)

    def write(self, data, lazy) -> None:
        raise NotImplementedError()


__SP_EXPORT__ = XMLFile
