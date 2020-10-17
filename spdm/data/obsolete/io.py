"""This module defines i/o operation of file."""
import collections
import contextlib
import pathlib
from urllib import parse

from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.urilib import urijoin, urisplit
from .connection import Connection, connect


def write(path, data, *args, **kwargs):
    with connect(path, *args, **kwargs) as conn:
        conn.write(data)


def read(path, *args, ** kwargs):
    with connect(path, *args,  **kwargs) as conn:
        res = conn.read()
    return res


def delete(path, *args, **kwargs):
    with connect(path, ) as conn:
        res = conn.delete(*args, ** kwargs)
    return res


# class DataPlugins(Plugins):
#     ''' IOPlugins
#         plugin example:

#         >>> import json
#             __plugin_spec__ = {"name": "json",  "url_pattern": ["*.json"]}
#             def load(fp):
#                 if isinstance(fp, str):
#                     fp = open(fp, "r")
#                 return json.load(fp)
#             def save(fp, d):
#                 if isinstance(fp, str):
#                     fp = open(fp, "w")
#                 return json.dump(d, fp)

#     '''

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _compile_url_pattern(self, url_p):
#         try:
#             return re.compile(url_p)
#         except re.error:
#             try:
#                 return re.compile(fnmatch.translate(url_p))
#             except re.error:
#                 raise ValueError(f"Illegal url pattern {url_p}")

#     def insert(self, mod):
#         if mod is None:
#             return False
#         elif not hasattr(mod, "load") or not hasattr(mod, "save"):
#             # logger.warning(f"Illegal IO plugin! {mod}")
#             return False

#         if hasattr(mod, "__plugin_spec__"):
#             pattern = mod.__plugin_spec__.get("url_pattern", None)
#             if pattern is None:
#                 pattern = []
#             elif not isinstance(pattern, collections.abc.Sequence):
#                 pattern = [pattern]

#             mod.__plugin_spec__["url_pattern"] = [
#                 self._compile_url_pattern(p) for p in pattern]

#         return super().insert(mod)

#     def pattern_match(self, url):
#         for m in self.values():
#             pattern = getattr(m, "__plugin_spec__", {"url_pattern": None})[
#                 "url_pattern"]
#             if pattern is None:
#                 continue
#             elif not isinstance(pattern, collections.abc.Sequence):
#                 logger.warning(
#                     f"__plugin_spec__.url_pattern should be a list {pattern}")
#                 continue

#             match = [p for p in pattern if hasattr(
#                 p, "fullmatch") and p.fullmatch(url)]
#             if len(match) > 0:
#                 return m

#         return None

#     def find(self, url):
#         s = parse.urlsplit(url)
#         return super().find(url) or super().find(s.scheme) \
#             or self.pattern_match(url)


# _plugins = DataPlugins(f"{__package__}.plugins", failsafe={"file": "json"})


# _server_plugins = Plugins(f"{__package__}.plugins.server")

# _server_plugins.insert(LocalFileDatabase)
# @functools.lru_cache(maxsize=100)
# def _connect_server(scheme, netloc):
#     """Connect to  database server or local filesystem.

#    * support:
#         * local : base on local filesystem
#         * mongodb
#         * mdsplus
#         * imas
#     todo:
#         * ssh
#         * postgresql
#         * h5srv

#     TODO (Salmon 2019.07.03) support combined scheme i.e. mongodb+hdf5,imas+uda
#     """
#     pass


# def connect(url,  scheme=None,   **kwargs):
#     """Connect server and fecth data.

#     url: <scheme>://<netloc>/<path>?<query>#<fragment>
#     * Uniform Resource Identifiers (URI): Generic Syntax
#         https://tools.ietf.org/html/rfc2396.html
#     * Relative Uniform Resource Locators
#         https://tools.ietf.org/html/rfc1808.html
#     Example:
#         <scheme>://<netloc> => Connection
#         <scheme>://<netloc>/<collection_path>
#             => connect(scheme,netloc).open(collection_path)
#         <scheme>://<netloc>/<collection_path>?<query>#<fragment>
#             => connect(scheme,netloc).open(collection_path)
#             .find(predict=query,projection=fragment)
#     """
#     spec = parse.urlparse(url)
#     path = pathlib.Path(spec.path)
#     prefix = path.parent

#     collection_name = path.name

#     logger.debug(spec)
#     logger.debug(path)
#     logger.debug(prefix)
#     logger.debug(collection_name)
#     if spec.scheme is None or spec.scheme == "":
#         conn = LocalFileDatabase.connect(
#             spec.netloc, prefix=prefix,   **kwargs)
#     else:
#         conn = _server_plugins.find(spec.scheme).connect(
#             spec.netloc,  prefix=prefix, **kwargs)

#     if conn is None:
#         raise ModuleNotFoundError(f"Can not load plugin for url '{url}'.")
#     if collection_name is '':
#         return conn
#     else:
#         collection = conn.open(collection_name)
#     if spec.query == "" and spec.fragment == "":
#         return collection
#     else:
#         return collection.find({k: v for k, v in parse.parse_qsl(spec.query)},
#                                spec.fragment)
