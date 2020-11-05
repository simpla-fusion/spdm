import collections
import os
import pathlib
import shlex
import subprocess
import traceback
import uuid

from .Factory import Factory
from .io import read, write
from .LazyProxy import LazyProxy
from .logger import logger
from .RefResolver import RefResolver
from .sp_export import sp_find_module
from .SpURI import urisplit, URISplitResult


class ModuleManager:
    def __init__(self, *args, prefix=None, suffix=".yaml", envs=None, **kwargs):
        self._resolver = None
        self._factory = None
        self._dirs = {}
        self._prefix = prefix or __package__.split('.')[0]
        self._suffix = suffix or ""

    def load_configure(self, path=None, **kwargs):
        if isinstance(path, (str, pathlib.Path)):
            path = [path]
        elif not isinstance(path, collections.abc.Sequence):
            raise TypeError(
                f"Type of path should be a string or  list of string!")

        conf = {}
        path.reverse()
        for p in path:
            merge_dict(conf, read(p))

        return conf

    def save_configure(self, path):
        raise NotImplementedError()

    def configure(self, conf=None, **kwargs):
        pkg_name = __package__.split('.', 1)[0]
        if conf is None:

            conf = self.load_configure(
                [f"pkgdata://{pkg_name}/../configure.yaml"])
        elif isinstance(self, collections.abc.Sequence):
            conf = self.load_configure(conf)
        elif not isinstance(conf, collections.abc.Mapping):
            raise TypeError(
                f"configure should be dict, string or list of string")

        other_conf = kwargs.get("configure", None) \
            or conf.get("configure", None)

        f_conf = collections.ChainMap(kwargs, conf, other_conf or {})

        self._resolver = RefResolver(**f_conf.get("resolver", {}))

        self._factory = Factory(**f_conf.get("factory", {}))

        repo_prefix = self._prefix.upper()

        self._home_dir = f_conf.get("home_dir", None) or \
            os.environ.get(f"{repo_prefix}_HOME_DIR", None)

        if self._home_dir is not None:
            self._home_dir = pathlib.Path(self._home_dir)
        else:
            self._home_dir = pathlib.Path.home()/f".{self._prefix.lower()}"

        self._modules_dir = self._home_dir / \
            f_conf.get(f"{repo_prefix}_MODULES_DIR", "modules")

        self._env_modules_dir = self._home_dir / \
            f_conf.get(f"{repo_prefix}_ENV_MODULEFILES_DIR", "env_modulefiles")

        self._software_dir = self._home_dir / \
            f_conf.get(f"{repo_prefix}_SOFTWARE_DIR", "software")

        self._sources_dir = self._home_dir / \
            f_conf.get(f"{repo_prefix}_SOURCES_DIR", "source")

        self._build_dir = self._home_dir / \
            f_conf.get(f"{repo_prefix}_BUILD_DIR", "build")

        self._resolver.alias.append(
            self._resolver.normalize_uri("/modules/*"),
            self._modules_dir.as_posix()+f"/*/{self._suffix}")

        self._conf = f_conf

    @property
    def dirs(self):
        return self._dirs

    @property
    def resolver(self):
        return self._resolver

    @property
    def factory(self):
        return self._factory

    def find_module(self, desc, version=None):
        if type(desc) is str:
            return self._resolver.fetch(f"{desc.replace('.', '/')}/{version or ''}")
        else:
            return self._resolver({**desc, "version": version})

    def new_class(self, desc):
        try:
            n_cls = self._factory.new_class(desc, _resolver=self._resolver)
        except Exception:
            raise ValueError(
                f"Can not make module {desc}! \n { traceback.format_exc()}")
        return n_cls

    def create(self, desc, *args, **kwargs):
        return self._factory.create(desc, *args, **kwargs)

    def glob(self, prefix=None):
        return self._resolver.glob(prefix or "/modules/")

    def install_spec(self, spec, no_build=False, force=False, **kwargs):
        ver = spec.get("version", "0.0.0")

        mid = spec.get("$id", None) or \
            kwargs.get("id", None) or \
            f"unkown/unamed_{uuid.uuid1().hex()}"

        spec["$id"] = mid

        spec_path = self._modules_dir/mid.replace('.', '/')/(ver+self._suffix)

        if not force and spec_path.exists():
            raise FileExistsError(f"'{spec_path}' exists!")

        spec_path.parent.mkdir(parents=True, exist_ok=force)

        write(spec_path, spec)

        if not no_build:
            self.build(spec_path, force=force)

    def install_from_repository(self, repo, *args, **kwargs):
        # TODO: clone repo to source_dir, find fy_module.yaml
        raise NotImplementedError()

    def install(self, spec, *args,   **kwargs):
        """ install specification
        """

        if isinstance(spec, (str, pathlib.Path, URISplitResult, collections.abc.Sequence)):
            spec = read(spec)

        if isinstance(spec, collections.abc.Mapping):
            return self.install_spec(spec, *args, **kwargs)
        else:
            raise TypeError(
                f"Need string or list of string or pathlib.Path. {type(spec)}")

    def build_repo(self, path, *args, **kwargs):
        raise NotImplementedError()

    def build(self, path,  *args, **kwargs):

        if isinstance(path, pathlib.Path) and path.is_dir():
            return self.build_repo(path, *args, **kwargs)

        spec = self._resolver.fetch(path)

        annotation = spec.get("annotation", {})

        eb_file = spec.get("build", {})

        eb_file["name"] = annotation.get("name", "unamed")
        eb_file["version"] = annotation.get("version", "0.0.0")
        eb_file["homepage"] = annotation.get("homepage", "")
        eb_file["description"] = annotation.get("description", "")

        eb_file["toolchain"]["version"]=eb_file["toolchain"]["version"].fomat(FY_TOOLCHAIN_VERSION=os.environ.get("FY_TOOLCHAIN_VERSION"))

        from env_modules_python import module

        module("load", build_cfg.get("eb_module", "EasyBuild"))

        # spec = DataEntry(self._get_module_spec_file(name)).read()

        # build_spec = spec.get("build", {})

        # build_ebfile = self._dirs["TEMP_DIR"]/f"{name.replace('.', '_')}.eb"

        # DataEntry(build_ebfile, format="yaml").write(build_spec)

        cmd = shlex.split(build_cfg.get("build_cmd", "eb"))

        try:
            logger.debug(f"Command:[ {' '.join(cmd)} ]")
            exit_status = subprocess.run(
                cmd, env=envs,
                capture_output=True,
                check=True,
                shell=False,
                text=True
            )
        except subprocess.CalledProcessError as error:
            logger.error(
                f"""Command failed! [{cmd}]
                   STDOUT:[{error.stdout}]
                   STDERR:[{error.stderr}]""")
            raise error

    def remove(self, name):
        raise NotImplementedError()
