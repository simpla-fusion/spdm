
import sys

from spdm.data.File import File
from spdm.common.logger import logger

if __name__ == '__main__':

    doc = File("/home/salmon/workspace/test_data/genray_profs_in.nc")

    logger.debug(doc.root["wz"])
