import pprint
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from spdm.data.File import File
from spdm.logger import logger
import sys


if __name__ == '__main__':
    doc = File(pathlib.Path(__file__).parent/"data/g063982.04800", format="GEQdsk")
    pprint.pprint(doc.root._holder)
