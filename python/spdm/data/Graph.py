'''
Graph
'''
import collections
import contextlib
import functools
import inspect

from ..util.LazyProxy import LazyProxy
from ..util.logger import logger
from ..util.SpObject import SpObject
from ..util.utilities import _empty

from .Edge import Edge
from .Node import Node
from .Group import Group
from .Port import InPort, OutPort, Port
from .Slot import InSlot, OutSlot, Slot


class Graph(Group):
    """Represents '''Graph'''.
        * defines namespace for the '''Node'''s
        * Graph is a Node

        TODO (salmon 2019.7.25): add subgraph
    """
    _stack = []

    def __init__(self,  *args, name=None, label=None, **kwargs):
        """Initialize graph."""
        super().__init__(*args, name=name, label=label or name,  **kwargs)
        self._closed = False
        self._children = {}

    ####################################################
    @classmethod
    def deserialize(cls, spec: collections.abc.Mapping, *args, **kwargs):
        instance = super().deserialize(spec,  *args, **kwargs)
        return instance

    def serialize(self):
        return {}

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self._name}' />"

    #########################################################################
    # as Context
    #

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._closed = True
        Graph._stack.pop()
        if len(Graph._stack) > 0:
            Node._current_graph = Graph._stack[-1]
        else:
            Node._current_graph = None

    def open(self):
        self._closed = False
        Graph._stack.append(self)
        Node._current_graph = Graph._stack[-1]
        return self

    #########################################################################
    # as Graph
    #
    def __len__(self):
        return len(self._children)

    @property
    def is_leaf(self):
        return len(self._children) == 0

    def node_add(self, nobj, *, enable_change_name=True, force=False, **kwargs):
        if not isinstance(nobj, Node):
            raise RuntimeWarning(
                f"Can not add/create child [{nobj.__class__}]! ")
        elif nobj._parent is not None and nobj._parent is not self:
            if not force:
                raise RuntimeError(f"Object '{nobj}' already has parent!")
            else:
                nobj.parent.remove_child(nobj)

        n_name = nobj.name

        if not enable_change_name and n_name in self._children:
            raise KeyError(f"Node {n_name} exists!")
        elif enable_change_name:
            num = functools.reduce(
                lambda prev, k: prev +
                (k == n_name or (
                    k is not None and k.startswith(f"{n_name}_"))),
                self._children.keys(), 0)

            if num > 0:
                n_name = f"{n_name}_{num}"

        self._children[n_name] = nobj
        # nobj._parent = self
        # for p in nobj._in_ports.values():
        #     if p._edge is None:
        #         continue
        #     p._edge = p._edge.split()

        return nobj, n_name

    def add_node(self, nobj, *args,  **kwargs):
        n, _ = self.node_add(nobj, *args, **kwargs)
        return n

    def register(self, nobj, *args,  **kwargs):
        _, k = self.node_add(nobj, *args, **kwargs)
        return k

    def remove_child(self, nobj, filter=None):
        # raise Warning("FUNCTION IS NOT COMPLETE!")
        if type(nobj) is str:
            pass
        elif isinstance(nobj, SpObject):
            nobj = nobj.name
        else:
            raise RuntimeError(f"Unknown object type [{type(nobj)}]")
        try:
            del self._children[nobj]
        except KeyError:
            logger.warning(f"Try to remove unexist key '{nobj}'!")

    def find_child(self, path, filter=None):
        if type(path) is str:
            path = path.split('.')
        obj = self
        for p in path:
            if not isinstance(obj, SpObject):
                break
            obj = obj._children.get(p, None)

        return obj

    @property
    def children(self):
        return self._children

    def has_child(self, n):
        res = False
        while isinstance(n, SpObject):
            res = n.parent is self
            if res:
                break
            else:
                n = n.parent
        return res

    def __contain__(self, n):
        if isinstance(n, Node):
            return self.has_child(n)
        elif isinstance(n, Edge):
            return self.has_child(n.source.parent) and self.has_child(n.target.parent)

    def has_node(self, n):
        return self.has_child(n)

    def create_node(self,  *args, **kwargs):
        """Create child 'cls' and add to self.
            Parameters
            ----------
            node_for_adding : node
                A node can be any hashable Python object except None.
            **kwargs : keyword arguments, optional
                Set or change node attributes using key=value.

            See Also
            --------
            add_nodes_from

            Examples
            --------
            >>> g = Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
            >>> n1=g.create_node([1])
            >>> n2=g.create_node(([1,,2,3],{"a":5}), name='Hello',)
        """
        return Node.create(*args, parent=self, **kwargs)

    def add_subgraph(self, *args,  **kwargs):
        return self.add_node(Graph(*args, parent=self, **kwargs))

    def find_node(self, *args, filter=None, **kwargs):
        return self.find_child(*args, filter=lambda n: (filter is None or filter(n)) and isinstance(n, Node), **kwargs)

    def remove_node(self, *args, filter=None, **kwargs):
        return self.remove_child(*args, filter=lambda n: (filter is None or filter(n)) and isinstance(n, Node), **kwargs)

    @property
    def nodes(self):
        return self._children.values()

    @property
    def node(self):
        return LazyProxy(self, None, {})

    def display(self, v):
        return Display(v, parent=self)

    def return_(self, v):
        s = self._children.get(f"_{Slot.TAG}_{OutPort.TAG}", None)

        if s is None:
            p = self.output
            s = self._children[f"_{Slot.TAG}_{OutPort.TAG}"]
        else:
            p = s.parent_port

        s.input.link(v)

        return p

    #################################################################

    @property
    def slot(self):
        """
            access or create in/out slots
            NOTE (salmon 2020520): Forbid access default input/output port throgh '''vars'''

            Example:
                >>>with Block(i=5) as b:
                        a= b.slot.i *2  # access input slot 'i'
                        b.slot.j="hello"      # create output slot 'j'
                        b.slot.i=b.slot.i+1 # create output slot 'i', and let slot_out_i = slot_in_i +1

                >>>print(b.result.j)
                hello
                >>>print(b.result.i)
                6

        """
        # if Node._current_graph is not self:
        #     raise RuntimeError(f" slot is only valid when its Group is Node._current_graph")

        def _internal_get(o, p):
            # check if var_k is change in block
            s = self._children.get(
                f"_{Slot.TAG}_{OutPort.TAG}_{p[0]}", None)
            if s is not None:
                return s.input.edge
            else:
                s = self._children.get(
                    f"_{Slot.TAG}_{InPort.TAG}_{p[0]}", None)
                if s is None:
                    raise KeyError(f"input slot '{p[0]}' is not defined!")
                return s.output

        def _internal_put(o, p, v):
            s = self._children.get(
                f"_{Slot.TAG}_{OutPort.TAG}_{p[0]}", None)
            if s is None:
                # port = self._out_ports.get(f"{OutPort.TAG}_{p[0]}", None) or \
                #     OutPort(name=f"{OutPort.TAG}_{p[0]}", label=p[0], parent=o)
                s = Slot(self._out_ports[f"{OutPort.TAG}_{p[0]}"], label="")

                s_in = self._children.get(
                    f"_{Slot.TAG}_{InPort.TAG}_{p[0]}", None)
                if s_in is not None:
                    s_in.loop_with(s.output)
                    s._label = s_in._label

            if isinstance(v, SpObject):
                s.input.link(v)
            else:
                s.input.link(Constant(v))

        return LazyProxy(self,  None,  level=0, get=_internal_get, put=_internal_put)

    @property
    def port(self):
        def _external_get(o, p):
            if p[0] == OutPort.TAG:
                pname = p[0]
            else:
                pname = f"{OutPort.TAG}_{p[0]}"
            s = o._children.get(f"_{Slot.TAG}_{pname}", None)
            if s is None:
                s = Slot(o._out_ports[pname], label="")
            return s.parent_port

        def _external_put(o, p, v):
            if p[0] == InPort.TAG:
                pname = p[0]
            else:
                pname = f"{InPort.TAG}_{p[0]}"
            s = o._children.get(f"_{Slot.TAG}_{pname}", None)
            if s is None:
                s = Slot(o._in_ports[pname], label="")
            s.parent_port.link(v)

        return LazyProxy(self,  None, level=0,
                         get=_external_get,
                         put=_external_put)

    @property
    def vars(self):
        if Node._current_graph is self:
            return self.slot
        else:
            return self.port

    @property
    def input(self):
        s = self._children.get(f"_{Slot.TAG}_{InPort.TAG}", None)
        if s is None:
            s = Slot(self._in_ports[InPort.TAG])
        return s.parent_port

    @property
    def output(self):
        s = self._children.get(f"_{Slot.TAG}_{OutPort.TAG}", None)
        if s is None:
            s = Slot(self._out_ports[OutPort.TAG], label="")
        return s.parent_port

    def bind(self, *args, **kwargs):
        for v in args:
            if not isinstance(v, SpObject):
                v = Constant(v)
            Slot(InPort(v, label="", parent=self), label="")

        for k, v in kwargs.items():
            if not isinstance(v, SpObject):
                v = Constant(v)
            Slot(InPort(v, name=f"{InPort.TAG}_{k}",
                        label=k, parent=self), label=k)

        return self

    ##############################################################################################
    # Control

    def preprocess(self, cache,  envs=None, *args, **kwargs):
        return super().preprocess(cache, envs, *args, **kwargs)

    def run(self, cache, *args, envs=None, **kwargs):
        return {s.parent_port.name: cache[s.id].get("value", _empty)
                for s in self._children.values() if isinstance(s, OutSlot)}

    def postprocess(self, cache,  envs=None, *args, **kwargs):
        return super().postprocess(cache, envs, *args, **kwargs)

    # def create_out_port(self, name, source=None):
    #     return self.add_node(OutSlot(name).bind(source)).output

    # def create_in_port(self, *args, **kwargs):
    #     return self.add_node(InSlot(*args, **kwargs)).output

    # def add_edge(self, source, target,  **edge_attrs):
    #     if self._closed:
    #         raise RuntimeError("Graph is closed!")
    #     if type(source) is str:
    #         source = self.find_child(source)
    #     if type(target) is str:
    #         target = self.find_child(target)

    #     if isinstance(source, Node):
    #         try:
    #             source = next(source.out_ports)
    #         except StopIteration:
    #             logger.debug(source, target)
    #             raise RuntimeError("Output port is not defined!")
    #     elif not isinstance(source, OutPort):
    #         source = Input(source, _parent=self).output

    #     if isinstance(target, Node):
    #         try:
    #             target = next(target.in_ports)
    #         except StopIteration:
    #             raise RuntimeError("Input port is not defined!")
    #     elif not isinstance(target, InPort):
    #         logger.debug(target)
    #         raise RuntimeError("Unknown input")
    #         # target = Output(target).input

    #     return target.link(source, **edge_attrs)

    # def find_edges(self, source, target):
    #     if type(target) is str:
    #         return self.find_edges(source, self.find_child(target))
    #     elif type(source) is str:
    #         return self.find_edges(self.find_child(source), target)
    #     elif isinstance(target, InPort):
    #         if target.edge is None:
    #             return []
    #         elif (isinstance(source, OutPort) and target.edge.source is source) or \
    #                 (isinstance(source, Node) and target.edge.source in source.out_ports):
    #             return [target.edge]
    #     elif isinstance(target, Node):
    #         e = [self.find_edges(source, p) for p in target.in_ports]
    #         return e if len(e) > 0 else []

    # def remove_edges(self, source, target):
    #     for e in self.find_edges(source, target):
    #         e.target.unlink()


__SP_EXPORT__ = Graph
# class FunctionWrapperGraph(Graph):
#     def __init__(self, func, *args, **kwargs):
#         super().__init__(*args, signature=inspect.signature(func), **kwargs)


# def as_graph(func=None, **attrs):
#     def _decorate(wrapped):
#         @functools.wraps(wrapped)
#         def _wrapper(*args, **kwargs):
#             obj = None
#             if inspect.isfunction(wrapped):
#                 obj = FunctionWrapperActor(wrapped,  **attrs)
#             else:
#                 raise ValueError(f"Can not convert {wrapped} to Actor!")
#             obj.bind(*args, **kwargs)
#             return obj
#         return _wrapper

#     if func is None:
#         return _decorate
#     else:
#         return _decorate(func)
