
from spdm.data import connect
from spdm.util.logger import logger
import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")
# import os

# os.environ['east_path'] = 'mds.ipp.ac.cn::/pcs_east'


if __name__ == '__main__':
    db = connect("imas://",
                 backend="mdsplus:///home/salmon/public_data/efit_east",  # "mdsplus://202.127.22.24/east_fit",
                 mapping_files=[
                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/static",
                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/dynamic"
                 ])

    entry = db.open(shot=55555).entry

    # logger.debug(entry.pf_active.coil[1].element[0].geometry.rectangle.r.__value__())
    # logger.debug(entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__())
    # logger.debug(entry.magnetics.b_field_tor_probe[1].field.data.__value__())

    logger.debug(entry.equilibrium.time_slice[1].profiles_2d.psi.__value__())

    # t = mds.Tree("east", 55555, mode="READONLY")
    # ip = t.getNode("\\ip")

    # entry.pf_active.coil[0].element[0].geometry.retangle.r = 5
    # logger.debug(entry.equilibrium.time_slice[0].profiles_2d[0].psi.__value__())
    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
