import collections.abc
import typing
from asyncio.log import logger
from copy import copy, deepcopy
import pprint
from ..common.tags import _undefined_
from .Query import Query

_TPath = typing.TypeVar("_TPath", bound="Path")


class Path(object):
    DELIMITER = '/'

    def __init__(self, *args):
        if len(args) == 1:
            self._items = Path._to_tuple(args[0])
        else:
            self._items = Path._to_tuple(args)

    def __repr__(self):
        return pprint.pformat(self._items)

    def __str__(self):
        return Path.DELIMITER.join([Path._to_str(d) for d in self._items])

    @staticmethod
    def _to_str(p: typing.Any) -> str:
        if isinstance(p, str):
            return p
        elif isinstance(p, slice):
            return f"{p.start}:{p.stop}:{p.step}"
        elif isinstance(p, int):
            return str(p)
        elif isinstance(p, collections.abc.Mapping):
            m_str = ','.join([f"{k}:{Path._to_str(v)}" for k, v in p.items()])
            return f"?{{{m_str}}}"
        elif isinstance(p, list):
            m_str = ','.join(map(Path._to_str, p))
            return f"[{m_str}]"
        elif isinstance(p, set):
            m_str = ','.join(map(Path._to_str, p))
            return f"{{{m_str}}}"
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {(p)}")

    @staticmethod
    def _from_str(v: str) -> typing.Any:
        v = v.strip(' ')

        if v.startswith('?'):
            res = Query(v)
        elif ':' in v:
            res = slice(*[int(s) for s in v.split(':')])
        elif v.startswith('[') and v.endswith(']'):
            res = [Path._to_tuple(s, force=False) for s in v[1:-1].split(',')]
        elif v.startswith('{') and v.endswith('}'):
            res = {Path._to_tuple(s, force=False) for s in v[1:-1].split(',')}
        elif v.isnumeric():
            res = int(v)
        else:
            res = v
        return res

    @staticmethod
    def _to_tuple(path, force=True) -> tuple:
        if path in (_undefined_,  None):
            res = tuple()
        elif isinstance(path, Path):
            res = deepcopy(path._items)
        elif isinstance(path, tuple):
            res = sum(map(Path._to_tuple, path), tuple())
        elif isinstance(path, str):
            p_list = path.split(Path.DELIMITER)
            if len(p_list) == 1:
                res = Path._from_str(p_list[0])
            else:
                res = tuple(map(Path._from_str, p_list))
        elif isinstance(path, (int, slice, Query)):
            res = path
        elif isinstance(path, set):
            res = {Path._to_tuple(item, False) for item in path}
        elif isinstance(path, collections.abc.Sequence):
            res = [Path._to_tuple(item, False) for item in path]
        elif isinstance(path, collections.abc.Mapping):
            res = Query(path)
        else:
            raise TypeError(f"Unknown Path type [{type(path)}]!")

        if force and not isinstance(res, tuple):
            res = (res,)
        return res

    @property
    def parent(self) -> _TPath:
        return Path(self._items[:-1])

    @property
    def items(self) -> tuple:
        return self._items

    def duplicate(self) -> _TPath:
        return Path(deepcopy(self._items))

    def append(self, *args) -> _TPath:
        return Path(self._items + Path._to_tuple(args))

    def __bool__(self) -> bool:
        return not self.empty

    def __len__(self):
        return len(self._items)

    def __iter__(self) -> None:
        yield from self._items

    def __truediv__(self, other: str) -> _TPath:
        return self.append(other)

    def __add__(self, other) -> _TPath:
        return self.append(other)

    def __iadd__(self, other) -> _TPath:
        raise NotImplementedError("Path should be immutable!")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Path(self._items[idx])
        else:
            return self._items[idx]

    def __setitem__(self, idx, item):
        raise NotImplementedError("Path should be immutable!")

    def __eq__(self, other: _TPath) -> bool:
        return self._items == other._items if isinstance(other,Path) else False

    @property
    def empty(self) -> bool:
        return len(self._items) == 0

    def as_list(self) -> list:
        return list(self._items)

    def join(self, delimiter=_undefined_) -> str:
        if delimiter is _undefined_:
            delimiter = self.DELIMITER
        return delimiter.join(map(str, self._items))

    # def split(self) -> list:
    #     if not isinstance(path, str):
    #         return path
    #     else:
    #         path = path.split(delimiter)
    #         return list(map(lambda p: int(p) if p.isnumeric() else p, path))

    def normalize(self) -> _TPath:
        if self._items is None:
            self._items = []
        elif isinstance(self._items, str):
            self._items = [self._items]
        elif isinstance(self._items, tuple):
            self._items = list(self._items)
        elif not isinstance(self._items, collections.abc.MutableSequence):
            self._items = [self._items]

        self._items = sum([d.split(Path.DELIMITER) if isinstance(d, str) else [d] for d in self._items], [])
        return self

    @property
    def is_closed(self) -> bool:
        return len(self._items) > 0 and self._items[-1] is None
