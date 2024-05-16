import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")
from ..util.logger import logger
from spdm.data import connect
import matplotlib.pyplot as pyplot

from spdm.core.plugins.PluginXML import open_xml

if __name__ == '__main__':

    doc = open_xml(["/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/static"])

    entry = doc.entry

    for coil in entry.pf_active.coil:
        logger.debug(coil.element.geometry.rectangle.__value__())
        break

    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
