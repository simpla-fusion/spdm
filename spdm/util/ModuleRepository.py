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
from .dict_util import format_string_recursive


class ModuleRepository:
    def __init__(self, *args, repo_name=None, repo_tag=None,  envs=None, module_file_suffix=None, **kwargs):
        self._factory = None
        self._repo_name = repo_name or __package__.split('.')[0]
        self._repo_tag = repo_tag or self._repo_name[:2]
        self._module_file_suffix = module_file_suffix or f"/{self._repo_tag.lower()}_module.yaml"
        self._envs = envs or {}

        logger.debug(f"Open module repository '{self._repo_name} ({self._repo_tag})'")

        self.configure([p for p in [
            *os.environ.get(f"{self._repo_name.upper()}_CONFIGURE_PATH", "").split(':'),
            *os.environ.get(f"{self._repo_tag.upper()}_CONFIGURE_PATH", "").split(':'),
            f"pkgdata://{self._repo_name}/configure.yaml"
        ] if not not p])

    def load_configure(self, path, **kwargs):
        return self.configure(io.read(path))

    def save_configure(self, path):
        io.write(path, self._envs)

    def configure(self, conf=None, **kwargs):
        logger.debug(conf)
        conf_path = []
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

        sys_conf = {} if not conf_path else io.read(conf_path)

        # TODO:  list in dict should be appended not overwrited .
        f_conf = collections.ChainMap(kwargs, conf, sys_conf)

        self._factory = Factory(**f_conf.get("factory", {}), resolver=RefResolver(**f_conf.get("resolver", {})))

        self._home_dir = f_conf.get("home_dir", None) or \
            os.environ.get(f"{self._repo_tag.upper()}_HOME_DIR", None)

        if self._home_dir is not None:
            self._home_dir = pathlib.Path(self._home_dir)
        else:
            self._home_dir = pathlib.Path.home()/f".{self._repo_name.lower()}"

        self._modules_dir = self._home_dir / \
            f_conf.get(f"{self._repo_tag.upper()}_MODULES_DIR", "modules")

        self._env_modules_dir = self._home_dir / \
            f_conf.get(f"{self._repo_tag.upper()}_ENV_MODULEFILES_DIR", "env_modulefiles")

        self._software_dir = self._home_dir / \
            f_conf.get(f"{self._repo_tag.upper()}_SOFTWARE_DIR", "software")

        self._sources_dir = self._home_dir / \
            f_conf.get(f"{self._repo_tag.upper()}_SOURCES_DIR", "source")

        self._build_dir = self._home_dir / \
            f_conf.get(f"{self._repo_tag.upper()}_BUILD_DIR", "build")

        self.resolver.alias.append(
            self.resolver.normalize_uri("/modules/*"),
            self._modules_dir.as_posix()+f"/*{self._module_file_suffix}")

        self._conf = f_conf

    @ property
    def resolver(self):
        return self._factory.resolver

    @ property
    def factory(self):
        return self._factory

    def find_description(self, desc, *args, version=None, tag=None, expand_template=True, **kwargs):
        """

        Format of searching path

            <desc>/<version>-<tag>

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
            res = self.resolver.fetch(collections.ChainMap(
                {"version": version, "tag": tag}, kwargs, desc))
        else:
            path = desc
            res = self.resolver.fetch(f"{path}/{version or 'default'}{tag or ''}")
            if not res:
                res = self.resolver.fetch(path)

        # if res is not None and not isinstance(res, AttributeTree):
        #     res = AttributeTree(res)

        if isinstance(res, collections.abc.Mapping) and expand_template:
            modulefile_path = pathlib.Path(res.get("$source_uri", ""))

            doc_vars = {k: v for k, v in res.items() if k.startswith('$') and isinstance(v, str)}
            envs = collections.ChainMap({
                "version": version or "",
                "tag": tag or "",
                "module_path": path,
                f"{self._repo_tag.upper()}_MODULEFILE_ROOT": modulefile_path.parent,
                f"{self._repo_tag.upper()}_MODULEFILE_ROOT": modulefile_path
            }, kwargs, doc_vars, self._envs)

            format_string_recursive(res,  envs)

        return res

    def new_class(self, desc, *args, **kwargs):

        desc_ = self.find_description(desc, *args, **kwargs)

        if not desc_:
            raise LookupError(f"Can not find description! {desc}")

        try:
            n_cls = self._factory.new_class(desc_)
        except Exception:
            raise ValueError(f"Can not make module {desc}! \n { traceback.format_exc()}")

        return n_cls

    def create(self, desc, *args, version=None, tag=None, **kwargs):
        n_cls = self.new_class(desc, version=version, tag=tag)

        if not n_cls:
            raise RuntimeError(f"Can not create object {desc} {n_cls}")

        return n_cls(*args, **kwargs)

    def glob(self, prefix=None):
        return self.resolver.glob(prefix or "/modules/")

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

        spec = self.resolver.fetch(path)

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
