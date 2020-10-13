from spdm.util.logger import logger
from spdm.data import connect
import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == '__main__':

    db = connect("hdf5:///home/salmon/workspace/output/east/test_")

    entry = db.insert(shot=55555).entry

    logger.debug(entry.pf_active.coil[1].element[0].geometry.rectangle.r.__value__())
    logger.debug(entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__())
    logger.debug(type(entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__()))

    entry.pf_active.coil[0].element[0].geometry.retangle.r=5
    
    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
