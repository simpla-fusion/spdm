import json
import pathlib
from urllib.request import urlopen

import requests
import yaml

from .sp_export import sp_pkg_data_path
from .SpURI import urisplit

FAIL_SAFE = None
ENABLE_REMOTE = False


def write(path, content, force=True):
    success = True
    o = urisplit(path)
    if o.schema in ["local", "file", None]:
        p = pathlib.Path(o.path)
        if not p.parent.exists():
            p.parent.mkdir(exists_ok=force, parents=True)
        with p.open(mode="w") as fid:
            if o.path.endswith(".yaml"):
                yaml.dump(content, fid,  Dumper=yaml.CDumper)
            elif o.path.endswith(".json"):
                json.dump(content, fid)
            else:
                success = False
    if not success:
        raise RuntimeError(f"Can not write  {path}")
    # else:
    #     raise NotImplementedError(f"Can not write {path}")


def read(uri):
    content = None
    o = urisplit(uri)
    if o.schema in [None, 'file', 'local']:
        p = pathlib.Path(o.path)
        if p.exists() and p.suffix in (".json", ".yaml", None):
            with p.open() as fid:
                content = yaml.load(fid, Loader=yaml.FullLoader)

    elif o.schema in ['pkgdata']:
        for p in sp_pkg_data_path(o.authority or __package__.split('.')[0], o.path[1:]):
            if p.exists() and p.suffix in (".json", ".yaml", ''):
                with p.open() as fid:
                    content = yaml.load(fid, Loader=yaml.FullLoader)
                    break
    elif o.schema in ['http', 'https'] and ENABLE_REMOTE:
        content = requests.get(uri).json()

    if content is None and FAIL_SAFE is not None:
        content = FAIL_SAFE(uri).read()
    return content


def glob(n_uri):

    o = urisplit(n_uri)
    path_list = []
    prefix, suffix = o.path.split('%_PATH_%')
    if o.schema in [None, 'file', 'local']:
        path_list = [prefix]
    elif o.schema in ['pkgdata']:
        path_list = [p for p in sp_pkg_data_path(
            o.authority or __package__.split('.', 1)[0], prefix)]

    for p in path_list:
        for f in pathlib.Path(p).rglob(f"**{suffix}"):
            fp = f.as_posix()
            name = fp[len(p):len(fp)-len(suffix)]
            yield name, f

    if len(path_list) == 0 and FAIL_SAFE is not None:
        handler = FAIL_SAFE(n_uri)
        if handler is not None:
            for p, f in handler.glob():
                yield p, f
