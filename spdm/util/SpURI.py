import collections
import copy
import pathlib
import re

from .logger import logger

_rfc3986 = re.compile(
    r"^((?P<schema>[^:/?#]+):)?(//(?P<authority>[^/?#]*))?(?P<path>[^?#]*)(\?(?P<query>[^#]*))?(#(?P<fragment>.*))?")


URISplitResult = collections.namedtuple(
    "URISplitResult", "schema authority path query fragment ")


def urisplit(uri):
    if isinstance(uri, URISplitResult):
        return uri
    elif uri is None:
        uri = ""
    elif isinstance(uri, pathlib.Path):
        uri = uri.as_uri()
    d = _rfc3986.match(uri).groupdict()
    d["schema"] = d.get("schema", "local")
    return URISplitResult(**d)


def uriunsplit(o):
    # if not isinstance(o, URISplitResult):
    #     raise TypeError(type(o).__name__)
    # return "".join([
    #     o.schema+"://" if o.schema is not None else "",
    #     (o.authority or "").strip('/'),
    #     o.path or "",
    #     "?"+o.query if o.query is not None else "",
    #     "#"+o.fragment if o.fragment is not None else ""
    # ])
    return uriunsplit2(**o._asdict())


def uriunsplit2(schema, authority, path, query=None, fragment=None):
    return "".join([
        schema+"://" if schema is not None else "",
        (authority or "").strip('/'),
        str(path) or "",
        "?"+query if query is not None else "",
        "#"+fragment if fragment is not None else ""
    ])


def urijoin(base, uri):
    o0 = urisplit(base)
    o1 = urisplit(uri)
    if o1.schema is not None and o1.schema != o0.schema:
        return uri
    elif o1.authority is not None and o1.authority != o0.authority:
        schema = o0.schema
        authority = o1.authority
        path = o1.path
    else:
        schema = o0.schema
        authority = o0.authority

        if o1.path is None or o1.path == '':
            path = o0.path
        elif o1.path is not None and len(o1.path) > 0 and o1.path[0] == '/':
            path = o1.path
        else:
            path = o0.path[:o0.path.rfind('/')]+"/"+o1.path

    return uriunsplit(URISplitResult(schema, authority, path, o1.query, o1.fragment))


def uridefrag(uri):
    o = urisplit(uri)
    return uriunsplit(URISplitResult(o.schema, o.authority, o.path, None, None)), o.fragment


class SpURI(object):
    def __init__(self, uri, *args, **kwargs):
        super().__init__()
        self.reset(uri)

    def reset(self, uri):
        if uri is None:
            pass
        elif isinstance(uri, pathlib.Path):
            uri = uri.as_uri()
        elif not isinstance(uri, str):
            raise TypeError(type(uri))

        o = urisplit(uri)
        self._schema = o.schema or "local"
        self._authority = o.authority or ""
        self._path = pathlib.Path(o.path)
        self._query = o.query
        self._fragment = o.fragment

    def copy(self):
        instance = object.__new__(SpURI)
        instance._schema = self._schema
        instance._authority = self._authority
        instance._query = self._query
        instance._fragment = self._fragment
        instance._path = copy.copy(self._path)
        return instance

    def __repr__(self):
        return self.as_uri()

    @property
    def schema(self):
        return self._schema

    @property
    def path(self):
        return self._path
    

    def as_key(self):
        return f"{self._schema}__{self._authority.replace('.','_')}_{self._path.as_key()}_{self._fragment.replace('.','_')}"

    def as_uri(self):
        return f"{self._schema}://{self._authority}{self._path.as_posix()}"

    def join(self, o_uri):
        n_uri = urijoin(self.as_uri(), o_uri)
        self._schema, self._authority, self._path, self._query, self._fragment = urisplit(
            n_uri)
        self._path = pathlib.Path(self._path)

    def as_short(self, length=20):
        s_path = self._path.as_posix()
        if len(s_path) > length:
            l_path = s_path.split("/")
            s_path = f"{l_path[0]}/{l_path[1]}/.../{l_path[-1]}"
        if self._schema in ['local', 'file']:
            return s_path
        else:
            return uriunsplit2(self._schema, self._authority, s_path, self._query, self._fragment)
