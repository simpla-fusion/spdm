import sys
import pathlib
from spdm.data.File import File
from spdm.util.logger import logger
import json
DATA_PATH = pathlib.Path(__file__).parent/"data"

if __name__ == '__main__':

    with File(DATA_PATH/"g063982.04800", mode="r", format="geqdsk") as fid:
        doc = fid.read()

    with File("../output/test.json", mode="w") as oid:
        oid.write(doc.dump())
        oid.close()
    with File("../output/test.json", mode="r") as oid:
        logger.debug(oid.read().dump())

    with File("../output/test.h5", mode="w") as oid:
        oid.write(doc.dump())
        oid.close()

    with File("../output/test.h5", mode="r") as oid:
        logger.debug(oid.read().dump())

    with File("../output/test.nc", mode="w") as oid:
        oid.write(doc.dump())
        oid.close()
    with File("../output/test.nc", mode="r") as oid:
        logger.debug(oid.read().dump())

    # with open("../output/test.json", mode="w") as fp:
    #     d = doc.dump(enable_ndarray=False)
    #     json.dump(d, fp)
