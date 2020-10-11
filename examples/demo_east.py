from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.data import connect

if __name__ == '__main__':

    collection = connect("imas+hdf5:///home/salmon/workspace/output/east/test_",
                         mapping_file="/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/config.xml")

    entry = collection.create(shot=55555)

    logger.debug(entry.pf_active.coil[1].element[0].geometry.rectangle.r)
    logger.debug(entry.wall.description_2d[0].limiter.unit[0].outline.r)
    logger.debug(type(entry.wall.description_2d[0].limiter.unit[0].outline.r.__fetch__()))

    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
