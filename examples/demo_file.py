
import sys

from spdm.data.File import File
from spdm.util.logger import logger

if __name__ == '__main__':

    doc = File(path="/home/salmon/workspace/output/physics_genray_10_13_200117_gompi_2020a_2/genray.nc")

    logger.debug(doc.root.entry.wz[:].__value__())
