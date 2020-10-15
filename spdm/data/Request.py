from spdm.util.LazyProxy import LazyProxy

# Request = collections.namedtuple("Request", "path query fragment")


class Request:
    DELIMITER = LazyProxy.DELIMITER

    def __init__(self, req="", *args, **kwargs):
        if isinstance(req, str):
            req = req.split(Request.DELIMITER)

        self._path = req
        self._query = {}

    @property
    def is_multiple(self):
        return False

    @property
    def path(self):
        return self._path.format_map(self._query)

    def apply(self, visitor):
        return visitor(self._path, self._query)

    @property
    def iter(self):
        yield self._path, self._query

    def append(self, seg):
        self._path(seg)

    def query(self, k, v):
        self._query[k] = v

    def add_slice(self, k, s):
        pass

    def __iter__(self):
        yield self._path, self._query
