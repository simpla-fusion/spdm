import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

from spdm.util.logger import logger
from spdm.data import connect


if __name__ == '__main__':

    db = connect("hdf5:///home/salmon/workspace/output/east/test_")

    entry = db.insert().entry

    entry.pf_active.coil[0].element[0].geometry.retangle.r = 5
    logger.debug(entry.pf_active.coil[0].element[0].geometry.retangle.r)
    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
