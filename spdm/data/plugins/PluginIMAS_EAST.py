import pathlib
from spdm.util.logger import logger
from ..Handler import HandlerProxy


def connect_imas_east(uri, *args, mapping_file=None, **kwargs):

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
        path = path.with_name(path.name+"{shot:08}_{run}.h5")
    else:
        path = path/"{shot:08}_{run}.h5"

    mapping_file = mapping_file or "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/"

    if isinstance(mapping_file, None) or isinstance(mapping_file, pathlib.Path):
        handler = HandlerProxy(MDSplusHandler(), mapping_file=mapping_file)
    else:
        handler = None

    return connect_mdsplus(uri, *args, id_pattern=filename_pattern, handler=handler, ** kwargs)


__SP_EXPORT__ = connect_imas_east
