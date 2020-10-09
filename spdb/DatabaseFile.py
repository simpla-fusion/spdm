import collections
import fnmatch
import os
import pathlib
import re
import inspect

from spdm.util.logger import logger

from spdm.util.utilities import getitem, whoami
from .Collection import Collection, DObject, ObjectId, oid
from .DataEntry import ContainerEntry, DataEntry


class FilePlugins(Plugins):
    ''' IOPlugins
        plugin example:

        >>> import json
            __plugin_spec__ = {
                "name": "json",
                "catalog":"data.server",
                "filename_pattern": ["*.json"]}
            def load(fp):
                if isinstance(fp, str):
                    fp = open(fp, "r")
                return json.load(fp)
            def save(fp, d):
                if isinstance(fp, str):
                    fp = open(fp, "w")
                return json.dump(d, fp)
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compile_filename_pattern(self, url_p):
        try:
            return re.compile(url_p)
        except re.error:
            try:
                return re.compile(fnmatch.translate(url_p))
            except re.error:
                raise ValueError(f"Illegal url pattern {url_p}")

    def insert(self, mod):
        if mod is None:
            return False
        elif not hasattr(mod, "load") or not hasattr(mod, "save"):
            # logger.warning(f"Illegal IO plugin! {mod}")
            return False
        name = self.get_plugin_name(mod)
        default_spec = {
            "name": name,
            "filename_pattern": [],
            "filename_extension": name
        }
        default_spec.update(getattr(mod, "__plugin_spec__", {}))

        pattern = default_spec["filename_pattern"]
        if pattern is None:
            pattern = []
        elif not isinstance(pattern, collections.abc.Sequence):
            pattern = [pattern]

        default_spec["filename_pattern"] = [
            self._compile_filename_pattern(p) for p in pattern]

        setattr(mod, "__plugin_spec__", default_spec)
        return super().insert(mod)

    def pattern_match(self, url):

        for m in self.values():
            pattern = m.__plugin_spec__["filename_pattern"]
            if pattern is None:
                continue
            elif not isinstance(pattern, collections.abc.Sequence):
                logger.warning(
                    f"__plugin_spec__.filename_pattern should be a list {pattern}")
                continue
            match = [p for p in pattern if hasattr(
                p, "fullmatch") and p.fullmatch(url)]
            if len(match) > 0:
                return m

        return None

    def find(self, url):
        return super().find(url) or self.pattern_match(url)


_file_plugins = FilePlugins(f"{__package__}.plugins.file", failsafe="json")


class LocalFileCollection(Collection):
    """Collection of local files."""

    def __init__(self, path: str,
                 schema: str = None,
                 load_file=None, save_file=None, spec=None,
                 backend=None,
                 prefix=None,
                 mode: str = "rw",
                 mask=0o755,
                 **kwargs):
        """Init."""
        super().__init__(schema=schema, **kwargs)
        self._mode = mode
        self._mask = mask

        self._filename_pattern = str(path)
        self._prefix = pathlib.Path.cwd() / prefix

        if inspect.ismodule(backend):
            mod = backend
        else:
            mod = _file_plugins[backend or self._filename_pattern]
        if mod is None:
            raise ModuleNotFoundError(
                f"Unable to determine the backend from path:{path}"
                "backend:{backend}")

        self._spec = spec or getattr(
            backend, "__plugin_spec__", {"name": schema})

        self._mod_load = (load_file or getattr(mod, "load", None))  \
            if 'r' in self._mode else None
        self._mod_save = (save_file or getattr(mod, "save", None)) \
            if 'w' in self._mode else None

        if '*' not in self._filename_pattern:
            self._filename_pattern = f"*.{self._spec['filename_extension']}"

        self._schema = schema

        def gen():
            return oid()
        self._next_id = gen

    def _file_path(self, idx: ObjectId):
        return self._prefix / (self._filename_pattern.replace("*", idx))

    def _load(self, fp, *args, **kwargs):
        if not self._mod_load:
            raise NotImplementedError(whoami(self))
        res = self._mod_load(fp, *args, **kwargs)
        if isinstance(res, DataEntry):
            return res
        else:
            return DataEntry(res)

    def _save(self, d, fp, *args, **kwargs):
        if not self._mod_save:
            raise NotImplementedError(whoami(self))
        if isinstance(fp, str):
            fp = pathlib.Path(fp)
        if not fp.parent.exists():
            fp.parent.mkdir(mode=self._mask, parents=True, exist_ok=True)
        if hasattr(d, "dump_data"):
            self._mod_save(d.data(), fp, *args, **kwargs)
        else:
            self._mod_save(d, fp, *args, **kwargs)

    def insert_one(self, document: DObject,  *args, **kwargs):
        idx = document.get("_id", None) or self._next_id()
        document.update({"_id": idx})
        self._save(document, self._file_path(idx), *args, **kwargs)
        return idx

    def insert_many(self, documents, *args, **kwargs) -> list:
        return [self.insert_one(doc, *args, **kwargs) for doc in documents]

    def find_one(self, predicate: DObject = None, projection: DObject = None):
        idx = getitem(predicate.get, "_id", None)
        flist = [self._file_path(idx)] \
            if idx is not None else self._prefix.glob(self._filename_pattern)
        res = None
        for fp in flist:
            res = self._load(fp).fetch_if(projection, predicate)
            if res is not None:
                break

        return res

    def find(self, predicate: DObject = None, projection: DObject = None):
        idx = predicate.get("_id", None) if predicate is not None else None
        if idx is None:
            logger.warning("NOT RECOMMENDED: Open many files!")
        for fp in self._prefix.glob(self._filename_pattern):
            if idx is not None and fp.stem != idx:
                continue
            else:
                res = self._load(fp).fetch_if(projection, predicate)
                if res is not None:
                    yield res

    def replace_one(self, predicate: DObject, replacement: DObject,
                    *args, **kwargs) -> ObjectId:
        idx = predicate.get("_id", None)

        flist = [self._file_path(idx)]  \
            if idx is not None else self._prefix.glob(self._filename_pattern)

        for fp in flist:
            if self._load(fp).check_if(predicate):
                fp.unlink()
                self._save(fp, replacement)
                return ObjectId(fp.stem)
        return None

    def update_one(self, predicate: DObject, update: DObject) -> ObjectId:
        idx = predicate.get("_id", None)
        flist = [self._file_path(idx)] \
            if idx is not None else self._prefix.glob(self._filename_pattern)

        for fp in flist:
            if self._load(fp).update_if(update, predicate):
                return fp.stem
        return None

    def update_many(self, predicate: DObject, update: DObject):
        idx = predicate.get("_id", None)
        if idx is None:
            logger.warning("NOT RECOMMENDED: Open many files !")
        for fp in self._prefix.glob(self._filename_pattern):
            if idx is not None and fp.stem != idx:
                continue
            elif self._load(fp).update_if(update, predicate):
                yield fp.stem

    def delete_one(self, predicate: DObject):
        idx = predicate.get("_id", None)
        flist = [self._file_path(idx)] \
            if idx is not None else self._prefix.glob(self._filename_pattern)

        for fp in flist:
            if self._load(fp).check_if(predicate):
                fp.unlink()
                return fp.stem
        return None

    def delete_many(self, predicate: DObject):
        idx = predicate.get("_id", None)

        for fp in self._prefix.glob(self._filename_pattern):
            if idx is not None and fp.stem != idx:
                continue
            elif self._load(fp).check_if(predicate):
                fp.unlink()
                yield fp.stem


class LocalFileDatabase(object):
    """Database of localfile."""

    __plugin_spec__ = {"name": "local"}

    def __init__(self, netloc, prefix=None, mode="rw", backend=None, **kwargs):
        super().__init__()
        # FIXME  netloc,prefix: resolve relative path
        self._mode = mode
        self._prefix = prefix
        self._backend = backend

    @classmethod
    def connect(cls,  *args, **kwargs):
        return cls(*args, **kwargs)

    def disconnect(self):
        self._prefix = None

    def open(self, ns_path, mode=None):
        # if self._prefix is None:
        #     raise ConnectionError(
        #         f"Database is not connected! {self.__class__.__name__}")
        return LocalFileCollection(ns_path,
                                   mode=mode or self._mode,
                                   prefix=self._prefix,
                                   backend=self._backend)


class LocalPath(SpBag):
    def __init__(self, path=None, *args, is_temp=False, **kwargs):
        self._is_temp = is_temp

        if is_temp and path is None:
            # TODO (salmon 20190829): temporary file or directory
            pass
        elif isinstance(path, str):
            self._path = pathlib.Path(path)
        elif isinstance(path, pathlib.Path):
            self._path = path
        elif path is None:
            raise ValueError(f"illegal  path! {path}")

        self._path = self._path.expanduser()
        logger.debug(self._path)

        super().__init__(*args, label=str(self._path), ** kwargs)

    @property
    def path(self):
        return self._path

    @property
    def is_temp(self):
        return self._is_temp

    # def __del__(self):
    #     if not self.is_temp:
    #         return
    #     elif self.is_file:
    #         self.unlink()
    #     elif self.is_dir():
    #         shutil.rmtree(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} path='{self._path.as_uri()}'/>"

    def fetch(self, session=None):
        if not self._path.exists():
            raise FileExistsError(f"Not exists {self._path.as_uri()}")

        return self._path.as_uri()


class File(SpBag, LocalPath):
    def __init__(self, path=None, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._plugin = _file_plugins.find(
            self.schema.get("file_type", None) or self.path)

    def read(self, *args, **kwargs):
        return self._plugin.load(self._path, *args, **kwargs)

    def write(self, d, *args, template=None, **kwargs):
        return self._plugin.save(d, self._path, *args,
                                 template=template or self.schema.get(
                                     "tempalte", None),
                                 **kwargs)


class Directory(bag.Group, LocalPath):
    @classmethod
    def make_bag(cls, fobj):
        if isinstance(fobj, LocalPath):
            return fobj
        elif isinstance(fobj, pathlib.Path):
            return LocalPath(fobj)
        elif not isinstance(fobj, collections.abc.Mapping):
            raise NotImplementedError()
        elif fobj.get("type", "File") == "File":
            return File(schema=fobj)
        else:
            return Directory(fobj)

    def __init__(self, path="./", *args,  **kwargs):
        super(Directory, LocalPath).__init__(path)
        super(Directory, bag.Group).__init__(*args,  **kwargs)
