import pathlib
from spdm.util.logger import logger
from .PluginHDF5 import connect_hdf5


def connect_imas_hdf5(uri, *args, **kwargs):

    def filename_pattern(p, d, mode=""):
        s = d.get("shot", 0)
        r = d.get("run", None)
        if mode == "glob":
            r = "*"
        elif(mode == "auto_inc"):
            r = len(list(p.parent.glob(p.name.format(shot=s, run="*"))))
        if r is None:
            return None
        else:
            return p.name.format(shot=s, run=str(r))

    path = pathlib.Path(getattr(uri, "path", uri))

    if path.suffix == "" and not path.is_dir():
        path=path.with_name(path.name+"{shot:08}_{run}.h5")
    else:
        path=path/"{shot:08}_{run}.h5"

    return connect_hdf5(path, *args, filename_pattern=filename_pattern, **kwargs)


__SP_EXPORT__=connect_imas_hdf5
