import collections
import os
import pathlib
import shlex
import subprocess
import traceback
import uuid

from . import io
from .Factory import Factory
from .LazyProxy import LazyProxy
from .logger import logger
from .RefResolver import RefResolver
from .sp_export import sp_find_module
from .SpURI import URISplitResult, urisplit
from .utilities import deep_update_dict
from .AttributeTree import AttributeTree
from .dict_util import tree_apply_recursive


class ModuleRepository:
    def __init__(self, *args, repo_prefix=None,  envs=None,  configure_file_suffix="/fy_module.yaml", **kwargs):
        self._resolver = None
        self._factory = None
        self._repo_prefix = repo_prefix or __package__.split('.')[0]
        self._configure_file_suffix = configure_file_suffix
        self._envs = envs or {}

    def load_configure(self, path, **kwargs):
        return self.configure(io.read(path))

    def save_configure(self, path):
        io.write(path, self._envs)

    def configure(self, conf=None, **kwargs):
        conf_path = [
            *os.environ.get(f"{self._repo_prefix.upper}_CONFIGURE_PATH", "").split(':'),
            f"pkgdata://{self._repo_prefix}/../configure.yaml"
        ]

        if isinstance(conf, str):
            conf_path.append(conf)
            conf = {}
        elif isinstance(conf, collections.abc.Sequence):
            conf_path.extend(conf)
            conf = {}
        elif isinstance(conf, collections.abc.Mapping):
            pass
        else:
            raise TypeError(f"configure should be dict, string or list of string")

        sys_conf = io.read(conf_path) if conf_path is not None else {}

        # TODO:  list in dict should be appended not overwrited .
        f_conf = collections.ChainMap(kwargs, conf, sys_conf)

        self._resolver = RefResolver(**f_conf.get("resolver", {}))

        self._factory = Factory(**f_conf.get("factory", {}), default_resolver=self._resolver)

        repo_prefix = self._repo_prefix.upper()

        self._home_dir = f_conf.get("home_dir", None) or \
            os.environ.get(f"{repo_prefix}_HOME_DIR", None)

        if self._home_dir is not None:
            self._home_dir = pathlib.Path(self._home_dir)
        else:
            self._home_dir = pathlib.Path.home()/f".{self._repo_prefix.lower()}"

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
            self._modules_dir.as_posix()+f"/*{self._configure_file_suffix}")

        self._conf = f_conf

    @property
    def resolver(self):
        return self._resolver

    @property
    def factory(self):
        return self._factory

    def find_description(self, desc, *args, version=None, tag_suffix=None, expand_template=True, **kwargs):
        """

        Format of searching path

            <desc>/<version>-<tag_suffix> 

            file link:
                <repo_path>/physics/genray/10.8.1   -> <repo_path>/physics/genray/10.8.1-foss-2019
                <repo_path>/physics/genray/10.8     -> <repo_path>/physics/genray/10.8.1-foss-2019
                <repo_path>/physics/genray/10       -> <repo_path>/physics/genray/10.8.1-foss-2019
                <repo_path>/physics/genray/default  -> <repo_path>/physics/genray/10.8.1-foss-2019

        Example:
                <repo_path>/physics/genray/10.8-foss-2019
                <repo_path>/physics/genray/10.8
                <repo_path>/physics/genray/default
                <repo_path>/physics/genray/

        """
        if type(desc) is not str:
            res = self._resolver.fetch(collections.ChainMap(
                {"version": version, "tag_suffix": tag_suffix}, kwargs, desc))
        else:
            path = desc
            res = None
            if tag_suffix is not None:
                res = self._resolver.fetch(f"{path}/{version or 'default'}-{tag_suffix}")
            elif version is not None:
                res = self._resolver.fetch(f"{path}/{version}")
            else:
                res = self._resolver.fetch(f"{path}/default")
                if not res:
                    res = self._resolver.fetch(path)

        # if res is not None and not isinstance(res, AttributeTree):
        #     res = AttributeTree(res)

        if isinstance(res, collections.abc.Mapping) and expand_template:
            envs = collections.ChainMap({}, self._envs)
            tree_apply_recursive(res, lambda s, _envs=envs: s.format_map(_envs), str)

        return res

    def new_class(self, desc, *args, **kwargs):

        desc_ = self.find_description(desc, *args, **kwargs)

        if not desc_:
            raise LookupError(f"Can not find description! {desc}")

        try:
            n_cls = self._factory.new_class(desc_, _resolver=self._resolver)
        except Exception:
            raise ValueError(f"Can not make module {desc}! \n { traceback.format_exc()}")

        return n_cls

    def create(self, desc, *args, version=None, tag_suffix=None, **kwargs):
        n_cls = self.new_class(desc, version=version, tag_suffix=tag_suffix)

        if not n_cls:
            raise RuntimeError(f"Can not create object {desc} {n_cls}")

        return n_cls(*args, **kwargs)

    def glob(self, prefix=None):
        return self._resolver.glob(prefix or "/modules/")

    def install_spec(self, spec, no_build=False, force=False, **kwargs):
        ver = spec.get("version", "0.0.0")

        mid = spec.get("$id", None) or kwargs.get("id", None) or f"unkown/unamed_{uuid.uuid1().hex()}"

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
            spec = io.read(spec)

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

        eb_file["toolchain"]["version"] = eb_file["toolchain"]["version"].fomat(
            FY_TOOLCHAIN_VERSION=os.environ.get("FY_TOOLCHAIN_VERSION"))

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
