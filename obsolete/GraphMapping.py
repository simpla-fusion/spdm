
import collections
import functools
import inspect
import networkx as spg
from .logger import logger
from .Signature import signature_from_spec
from .RefResolver import RefResolver


class GraphMapping(spg.DiGraph):
    def __init__(self, *args,  spec_prefix=None, schema="Mapping", resolver=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = spec_prefix or "mapping"
        self._resolver = resolver or RefResolver()

    @property
    def resolver(self):
        return self._resolver

    def add_edge(self, source, target, **kwargs):
        source = self.resolver.remove_prefix(source)
        target = self.resolver.remove_prefix(target)
        return super().add_edge(source, target, **kwargs)

    def from_chain(self, chain):
        logger.debug(chain)
        return lambda p: p

    def from_spec(self, spec):
        return self._mapper_from_spec

    def find_mapper(self, source: str, target: str):
        source = self._resolver.remove_prefix(source)
        target = self._resolver.remove_prefix(target)

        m = self.get_edge_data(source, target, {}).get("mapper", None)

        if m is None:
            m = self._resolver.resolve(f"{self._prefix}/{source}/{target}")
            if m is not None:
                self.add_edge(source, target, mapper=m)
            else:
                try:
                    m = self.from_chain(
                        spg.shortest_path(self, source, target))
                except IndexError:
                    raise RuntimeError(
                        "Can not find route from {source} to {target}!")

        return m

    def map(self, v, schema):
        pass
