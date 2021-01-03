import collections
import copy
import re
import pathlib
from .logger import logger


_r_path_item = re.compile(
    r"([a-zA-Z_\$][^/\\\[\]]*)|\[([+-]?\d*)(?::([+-]?\d*)(?::([+-]?\d*))?)?\]")


class _Empty:
    pass


def _int_or_none(v):
    """if v is not None return int(v) else return None"""
    return int(v) if v is not None and v != '' else None


def _path_string_as_iter(path, with_position=False):
    """ obj : object like dict or list
        path:
            i.e.  a.b.d[23][3:3:4].adf[3]
        try_attr: if true then try to get attribute  when getitem failed

    """

    for m in _r_path_item.finditer(path):
        attr, start, stop, step = m.groups()
        if attr is not None:
            idx = attr
        elif stop is None and step is None:
            idx = _int_or_none(start)
        else:
            idx = slice(_int_or_none(start), _int_or_none(
                stop), _int_or_none(step))

        if with_position:
            yield idx, m.end()
        else:
            yield idx


def _join_path(path, *path_parts):

    if len(path_parts) == 0:
        return

    for p in path_parts:
        if p is None:
            continue
        elif isinstance(p, str):
            if p[0] == '/':
                path.clear()
                p = p[1:]
            path.extend([v for v in _path_string_as_iter(p)])

        # elif isinstance(p, int) or isinstance(p, slice):
        #     path.append(p)
        # elif isinstance(path_parts, collections.abc.Iterator) or isinstance(path_parts, collections.abc.Sequence):
        #     _join_path(path, p)
        else:
            path.append(p)
            # logger.warning(
            #     f"Illegal type '{ TypeError(type(p).__name__)}' is ignored! ")


class SpPath(object):

    __slots__ = "_root", '_path'

    def __init__(self,  path=None, root=None, *args, **kwargs):
        super().__init__()
        self._root = root
        self._path = []
        self.join(path)

    def clone(self):
        instance = self.__class__.__new__(self.__class__)
        instance._root = self._root
        instance._path = copy.copy(self._path)
        return instance

    def rebase(self, new_root):
        instance = self.clone()
        instance._root = new_root
        return instance


    @property
    def root(self):
        return self._root

    @property
    def path(self):
        return pathlib.Path("".join(map(lambda s: f"/{s}" if isinstance(s, str) else f"[{s}]", self._path)))

    @property
    def name(self):
        return self.path.name

    def __truediv__(self, r_path: str):
        if r_path is None:
            return self.clone()
        else:
            return self.clone().join(r_path)

    def __getitem__(self, idx):
        return self.clone().join(idx)

    def join(self, *path_parts):
        _join_path(self._path, *path_parts)
        return self

    def extend(self,  other_path):
        if other_path is None:
            raise ValueError(f"Insert NONE path")
        self._path.extend(other_path)
        return self

    def as_key(self):
        def _quote(s):
            if isinstance(s, str):
                return s
            elif isinstance(s, slice):
                return f"__{s.start}_{s.stop}_{s.step}__"
            elif isinstance(s, collections.abc.Sequence):
                return "_".join([_quote(v) for v in s])
            else:
                return str(s)
        return ("_".join(map(_quote, self._path))).replace(" ", "_")

    def as_uri(self):
        def _quote(s):
            if isinstance(s, str):
                return s
            elif isinstance(s, slice):
                s = f"{s.start or ''}:{s.stop or ''}:{s.step or ''}"
                if s.endswith("::"):
                    s = s.replace("::", ":")
                return s
            elif isinstance(s, list):
                return "["+",".join([_quote(v) for v in s])+"]"
            elif isinstance(s, tuple):
                return ",".join([_quote(v) for v in s])
            elif not isinstance(s, str):
                return str(s)

        uri = (
            "".join(map(lambda s: f"/{_quote(s)}" if isinstance(s, str) else f"[{_quote(s)}]", self._path)))
        # if len(uri) > 0 and uri[0] == '/':
        #     uri = uri[1:]
        return uri

    def as_short(self, length=20):
        # message = self.as_uri()
        # if len(message) > length:
        #     o = message.rsplit("/", 1)

        #     if len(o) < 2:
        #         message = "..."+message[-10:]
        #     else:
        #         message = ".../"+o[1]
        # return message
        if len(self._path) < 5:
            return "/".join(self._path)
        else:
            return f"{self._path[0]}/{self._path[1]}/.../{self._path[-1]}"
