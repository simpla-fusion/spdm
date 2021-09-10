import sys
import pathlib
from spdm.data.File import File
from spdm.util.logger import logger

DATA_PATH = pathlib.Path(__file__).parent/"data"

if __name__ == '__main__':

    with File(DATA_PATH/"g063982.04800", mode="r", format="geqdsk") as fid:
        doc = fid.read()
        logger.debug(doc.dump())
