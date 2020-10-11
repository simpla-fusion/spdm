from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.data import connect

if __name__ == '__main__':

    def filename_pattern(p, d, auto_inc=False):
        try:
            fname = p.name.format_map(d)
        except KeyError:
            s = d.get("shot", 0)
            r = d.get("run", None)
            if r is None and auto_inc:
                logger.debug(p.name.format(shot=s, run="*"))
                r = len(list(p.parent.glob(p.name.format(shot=s, run="*"))))
            fname = p.name.format(shot=s, run=str(r))

        return fname
    collection = connect("hdf5:///home/salmon/workspace/output/east_{shot:08}_{run}.h5",
                         filename_pattern=filename_pattern)

    entry = collection.create(shot=55555)
    # entry1 = collection.open(_id=55555, run=5)
    # entry.a = 5
    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
