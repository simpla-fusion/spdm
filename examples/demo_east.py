from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.data import connect

if __name__ == '__main__':

    collection = connect("imas+hdf5:///home/salmon/workspace/output/east/test_")

    entry = collection.create(shot=55555)
    # entry1 = collection.open(shot=55555)
    entry.a.b.c = [{"e": 5}, {"d": [1, 4, 3], "f":"hello world"}]
    entry.a.b.__a = 5
    logger.debug(entry.a.b.c)

    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
