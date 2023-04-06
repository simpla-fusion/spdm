from __future__ import annotations

import collections.abc
import pprint
import typing
from copy import deepcopy

from ..common.tags import _undefined_

_TKey = typing.TypeVar('_TKey', int, str)
_TIndex = typing.TypeVar('_TIndex', int, slice, str, typing.Sequence, typing.Mapping)


class Path(list):

    DELIMITER = '/'

    def __init__(self, *args):
        super().__init__(sum(map(Path._normalize, args), list()))

    def __repr__(self):
        return pprint.pformat(self)

    def __str__(self):
        return Path.DELIMITER.join([Path._to_str(d) for d in self])

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    @property
    def parent(self) -> Path:
        other = self.duplicate()
        if len(other) == 0:
            raise RuntimeError()
        else:
            other.pop()
        return other

    @property
    def is_closed(self) -> bool:
        return len(self) > 0 and self[-1] is None

    def close(self) -> Path:
        if not self.is_closed:
            self.append(None)
        return self

    def open(self) -> Path:
        if self[-1] is None:
            self.pop()
        return self

    def duplicate(self, new_value=None) -> typing.Any:
        # other = object.__new__(self.__class__)
        return Path(self[:] if new_value is None else (new_value))

    def as_query(self) -> typing.Tuple[typing.Union[Path, None], typing.Any]:
        """
        将path转换为query 和 predicate
        """
        # typing.Union[Path, typing.Dict[str, Path], typing.List[Path]]:
        # 一句命令从self中找到第一个不是int和str的元素，返回index
        return self._as_query(self)

    def _as_query(self, p: typing.List[typing.Any]) -> typing.Any:  # Tuple[typing.Union[Path, None], typing.Any]:
        indcies = {i: v for i, v in enumerate(p) if not isinstance(v, (int, str))}
        pos = 0
        query = []
        for idx, pred in indcies.items():
            query.append((p[pos: idx-1], pred))
            pos = idx+1

        if index is None:  # 如果index为None，说明p中只有str或者int，直接返回
            return Path(p), None
        elif index > 0:  # 如果index>0，说明p中有str或者int，需要先处理这部分
            return Path(p[:index-1]), self._as_query(p[index:])[1]    # 递归处理
        # 如果index=0，说明p中第一个元素就是需要处理的元素
        elif isinstance(p[index], set):
            return None, {k: self._as_query([k]+p[index + 1:]) for k in p[index]}
        elif isinstance(p[index], collections.abc.Mapping):
            return None, {k1: self._as_query([k2]+p[index + 1:]) for k1, k2 in p[index].items()}
        elif isinstance(p[index], collections.abc.Sequence):
            return None, [self._as_query([k]+p[index + 1:]) for k in p[index]]
        elif isinstance(p[index], slice):
            start, stop, stride = p[index].start, p[index].stop, p[index].step
            if stop is None or (stop-start)*stride < 0:
                raise IndexError(f"Cannot slice with stop=None, {p[index]}")
            return None, [self._as_query([k]+p[index + 1:]) for k in range(start, stop, stride)]
        else:
            raise NotImplementedError(f"Not support Query,list,mapping,tuple to str,yet! {p[index]}")

    def append(self, *args) -> Path:
        if self.is_closed:
            raise ValueError(f"Cannot append to a closed path {self}")
        self += Path._normalize(*args)
        return self

    def __bool__(self) -> bool:
        return len(self) > 0

    def __truediv__(self, p) -> Path:
        return self.duplicate().append(p)

    def __add__(self, p) -> Path:
        return self.duplicate().append(p)

    def __iadd__(self, p) -> Path:
        return self.append(p)

    def __eq__(self, other) -> bool:
        if isinstance(other, list):
            return super().__eq__(other)
        elif isinstance(other, Path):
            return super().__eq__(other[:])
        else:
            return False

    def join(self, delimiter: str = "") -> str:
        if delimiter == "":
            delimiter = self.DELIMITER
        return delimiter.join(map(str, self))

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
            res = [Path._normalize(s) for s in v[1:-1].split(',')]
        elif v.startswith('{') and v.endswith('}'):
            res = {Path._normalize(s) for s in v[1:-1].split(',')}
        elif v.isnumeric():
            res = int(v)
        else:
            res = v
        return res

    @staticmethod
    def _normalize(path) -> typing.Any:
        if path in (_undefined_,  None):
            res = list()
        elif isinstance(path, Path):
            res = deepcopy(path[:])
        elif isinstance(path, list):
            res = list(map(Path._normalize, path))
        elif isinstance(path, str):
            res = list(map(Path._from_str, path.split(Path.DELIMITER)))
        elif isinstance(path, (int, slice, dict)):
            res = [path]
        elif isinstance(path, set):
            res = [{Path._normalize(item) for item in path}]
        elif isinstance(path, collections.abc.Sequence):
            res = [(Path._normalize(item) for item in path)]
        elif isinstance(path, collections.abc.Mapping):
            res = [{k: Path._normalize(v) for k, v in path.items()}]
        else:
            raise TypeError(f"Unknown Path type [{type(path)}]!")

        return res
