
import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")
import spdm.data as spdb
from spdm.util.logger import logger

if __name__ == '__main__':

    # db = spdb.Collection("hdf5:///home/salmon/workspace/output/east/test_")

    # entry = db.insert().entry

    doc = spdb.Document("/home/salmon/workspace/output/east/test_a.h5", mode="w")

    doc.copy({
        "a": {"b": [1, 2, 3, 4]}
    })

    logger.debug(doc.root.entry)
    # entry = doc.entry

    # entry.pf_active.coil[0].element[0].geometry.retangle.r = 5
    # logger.debug(entry.pf_active.coil[0].element[0].geometry.retangle.r)
    # pprint.pprint(collection)
    # a = entry.pf_active.coil[0].element[0].geometry.retangle.r
    # b = a.__fetch__()
    # logger.debug(type(a))
    # logger.debug(type(b))
