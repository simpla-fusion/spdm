from __future__ import annotations
import collections
import inspect
import abc
import typing
import math
import dataclasses
from .Path import Path, as_path
from .HTree import HTree, HTreeNode
from .Expression import Expression
from .sp_property import PropertyTree
from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.typing import array_type


class Edge:
    """`Edge` defines a connection between two `Port`s 
    
    Attribute

    - source      : the start of edge which must be `OUTPUT Port`
    - target      : the start of edge which must be `INPUT Port`
    - dtype       : defines what `Port`s it can be connected, (default: string)
    - label       : short string
    - description : long string
    """

    class Endpoint:
        def __init__(self, node, type_hint=None) -> None:
            if isinstance(node, Edge.Endpoint):
                type_hint = type_hint or node.type_hint
                node = node.node

            self.node: HTreeNode = None

            self.type_hint: typing.Type = None

            self._time: float = None

            self._iteration: int = None

            # self.update(node, type_hint)

        def update(self, node=None, type_hint=None) -> Edge.Endpoint:
            if type_hint is not None and type_hint is not _not_found_:
                self.type_hint = type_hint

            if node is _not_found_ or node is None or node is self.node:
                pass

            elif not inspect.isclass(self.type_hint) or isinstance(node, self.type_hint):
                self.node = node
            else:
                raise TypeError(f"{node} is not {self.type_hint}")

            self._time = getattr(self.node, "time", -math.inf)
            self._iteration = getattr(self.node, "iteration", -1)

        def unlink(self):
            self.node = None

        def __copy__(self):
            return Edge.Endpoint(self.node, self.type_hint)

        @property
        def is_changed(self) -> bool:
            return not (
                (not hasattr(self.node, "time") or math.isclose(self.node.time, self._time))
                and (not hasattr(self.node, "iteration") or self.node.iteration == self._iteration)
            )

    def __init__(
        self,
        source=None,
        target=None,
        source_type_hint=None,
        target_type_hint=None,
        graph=None,
        **kwargs,
    ):
        self._source = Edge.Endpoint(source, source_type_hint)

        self._target = Edge.Endpoint(target, target_type_hint)

        self._graph = graph

        self._metadata = PropertyTree(kwargs)

    def __copy__(self):
        return Edge(self._source, self._target, graph=self._graph, **self._metadata._cache)

    @property
    def metadata(self) -> PropertyTree:
        return self._metadata

    @property
    def source(self) -> Endpoint:
        return self._source

    @property
    def target(self) -> Endpoint:
        return self._target

    @property
    def is_linked(self):
        return self._target.node is not None and self._source.node is not None

    def __str__(self):
        return f"""<{self.__class__.__name__} source='{self._source}' target='{self._target}' label='{self._metadata.get('label','')}'/>"""

    def _repr_s(self):
        def _str(s):
            if s is None:
                return ""
            elif type(s) is str:
                return "." + s
            elif isinstance(s, slice):
                if s.stop is None:
                    return f"[{s.start or ''}:{s.step or ''}]"
                else:
                    return f"[{s.start or ''}:{s.step or ''}:{s.stop or ''}]"

            elif isinstance(s, collections.abc.Sequence):
                return "".join([f"{_str(t)}" for t in s])
            else:
                return f"[{s}]"

        return "".join([_str(s) for s in self._path])

    def split(self, *args, **kwargs):
        """
        using Slot Node split edge into chain, add In(Out)Slot not to graph

        return list of splitted edges
        """
        source = self._source
        target = self._target

        s = collections.deque([source.parent] if source is not None else [])
        t = collections.deque([target.parent] if target is not None else [])

        if getattr(s[0], "parent", None) is getattr(t[0], "parent", None):
            return self

        while s[0] is not None:
            s.appendleft(s[0].parent)

        while t[0] is not None:
            t.appendleft(t[0].parent)

        s_rank = len(s)
        t_rank = len(t)
        pos = s_rank - 2
        tag = ""
        while pos >= t_rank or (pos >= 0 and s[pos] is not t[pos]):
            tag = f"{tag}_{source.parent.name}"

            s[pos].slot[tag] = source

            source = s[pos].port[tag]

            pos = pos - 1

        tag = f"{tag}_{source.parent.name}"

        while pos < t_rank - 2:
            pos = pos + 1
            tag = f"{tag}_{source.parent.parent.name}"
            t[pos].port[tag] = source

            source = t[pos].slot[tag]

        src, prefix = self._unwrap(source)
        self._source = src
        self._path = prefix + self._path
        return self


class Ports(typing.Dict[str, Edge]):
    def __init__(self, holder):
        self._holder = holder

    @abc.abstractmethod
    def link(self, id, node, type_hint=None) -> Edge:
        raise NotImplementedError(f"This is an abstract method!")

    def fetch(self) -> typing.Dict[int | str, typing.Any]:
        return {k: e.source.node for k, e in self.items()}

    def refresh(self):
        return True

    def get_source(self, key, default_value=_not_found_):
        obj = super().get(key, _not_found_)
        if not isinstance(obj, Edge) or obj.source.node is None:
            return default_value
        else:
            return obj.source.node

    def get_target(self, key, default_value=_not_found_):
        obj = super().get(key, _not_found_)
        if not isinstance(obj, Edge) or obj.target.node is None:
            return default_value
        else:
            return obj.target.node


class InPorts(Ports):
    def __missing__(self, name: str | int) -> Edge:
        return self.setdefault(name, Edge(None, self._holder))

    def link(self, id, source, type_hint=None):
        edge = self[id]
        edge.source.update(source, type_hint)
        return edge

    def update(self, kwargs: typing.Dict[str, typing.Any]):
        for k, v in kwargs.items():
            self[k].source.update(v)


class OutPorts(Ports):
    def __missing__(self, name: str | int) -> Edge:
        return self.setdefault(name, Edge(self._holder, None))

    def link(self, id, target, type_hint=None) -> Edge:
        edge = self[id]
        edge.target.update(target, type_hint)
        return edge

    def update(self, kwargs: typing.Dict[str, typing.Any]):
        for k, v in kwargs.items():
            self[k].target.update(v)

    def set(self, key, value):
        self[key].target.node = value
