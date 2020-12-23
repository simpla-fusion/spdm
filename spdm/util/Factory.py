import collections
import copy
import functools
import inspect
import pprint

from .Alias import Alias
from .logger import logger
from .sp_export import sp_find_module
from .urilib import urijoin, urisplit
from .utilities import iteritems, whoami
from .SpObject import SpObject


class Factory(object):
    """ Convert schema to class
    """

    def __init__(self, *args, default_handler=None, alias=None, default_resolver=None, ** kwargs):
        super().__init__()
        self._alias = Alias(glob_pattern_char="*")
        self._handlers = {}
        self._cache = {}
        self._alias.append_many(alias)
        self._default_resolver = default_resolver

    @property
    def alias(self):
        return self._alias

    def add_alias(self, s, t):
        if self._default_resolver is not None:
            s = self._default_resolver.normalize_uri(s)
        self._alias.append(s, t)

    @property
    def cache(self):
        return self._cache

    def create(self, desc, *args, _resolver=None, **kwargs):
        """ Create a new instance """
        n_cls = self.new_class(desc,  resolver=_resolver)
        if not n_cls:
            raise RuntimeError(f"Create cls failed! {desc}")
        return n_cls(*args, **kwargs)

    def new_class(self, desc, *args, _resolver=None, **kwargs):
        """ Create a new class from description
            Parameters:
                desc: description of class
        """
        if _resolver is None:
            _resolver = self._default_resolver

        if type(desc) is str:
            c_id = desc if _resolver is None else _resolver.normalize_uri(desc)
            n_cls = self._cache.get(c_id, None)
            if n_cls is not None:
                return n_cls

        if _resolver is not None:
            desc = _resolver.fetch(desc)

        if not isinstance(desc, collections.abc.Mapping):
            raise ValueError(f"Can not resolve schema {desc}!")

        n_cls = None

        # search handler
        for h_req in self._alias.match(desc.get("$id", None),
                                       desc.get("$base", None),
                                       desc.get("$schema", None)):
            n_cls = self.handle(h_req, desc, *args, _resolver=_resolver, **kwargs)
            if n_cls is not None:
                break

        if n_cls is not None:
            self._cache[desc["$id"]] = n_cls

        return n_cls

    def handle(self, uri, *args, **kwargs):
        o = urisplit(uri)
        h = getattr(self, f"_handle_{o.schema}", None) or self._handlers.get(o.schema, None)
        if h is None:
            raise LookupError(f"Can nod handle {uri}")
        return h(uri, *args, **kwargs)

    def _handle_PyObject(self, uri, desc, *args, resolver=None, ** kwargs):
        o = urisplit(uri)
        assert(o.schema == 'PyObject')
        if o.authority is None or o.authority == '':
            handler = sp_find_module(o.path, o.fragment)
        else:
            fragment = '/'+o.fragment if o.fragment is not None else ''
            handler = sp_find_module(o.authority, f"{o.path}{fragment}")

        if issubclass(handler, SpObject):
            n_cls = handler.new_class(desc, *args,  factory=self, resolver=self._default_resolver, **kwargs)
        elif inspect.isclass(handler) and hasattr(handler, "new_class"):
            n_cls = handler.new_class(desc, *args, _factory=self, **kwargs)
        elif not inspect.isclass(handler) and callable(handler):
            n_cls = handler(desc, *args, _factory=self, **kwargs)

        return n_cls

    def _handle_SpModule(self, uri, desc,    **kwargs):
        # handler_path = schema.get('$schema')
        # handler = None
        # if handler_path is not None:
        #     o = urisplit(handler_path)
        #     handler = sp_find_module(o.path, o.fragment)
        # if handler is None:
        #     raise ModuleNotFoundError(f"Can't find handler {schema}")
        # n_class = None
        # if hasattr(handler, "new_class"):
        #     n_class = handler.new_class(
        #         schema=schema, factory=factory, **kwargs)
        # elif callable(handler):
        #     n_class = handler(schema=schema, factory=factory, **kwargs)
        # return n_class
        raise NotImplementedError()

    # def expand(self, spec, level_of_expanding=0):
    #     if isinstance(spec, collections.abc.Mapping) and "$schema" in spec:
    #         return self.create(spec)
    #     elif level_of_expanding <= 0:
    #         return spec
    #     elif isinstance(spec, collections.abc.Mapping):
    #         return {k: self.expand(v, level_of_expanding-1)
    #                 if k[0] != '@' and k[0] != '$' else v for k, v in spec.items()}
    #     elif not isinstance(spec, str) and isinstance(spec, collections.abc.Sequence):
    #         return [self.expand(v, level_of_expanding-1) for v in spec]
    #     else:
    #         return spec

    # def validate(self, spec, schema=None):
    #     if self._validater is None:
    #         # logger.warning("Validator is not defined!")
    #         return

    #     if isinstance(spec, collections.abc.Mapping):
    #         schema = schema or spec.get("$schema", None)

    #     if isinstance(schema, str):
    #         schema = {"$ref": schema}
    #     try:
    #         self._validater.validate(spec, schema)
    #     except Exception as error:
    #         raise error
    #     else:
    #         logger.info(f"Validate schema '{spec.get('$schema')}' ")
