import sys

sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


from spdm.util.logger import logger
from spdm.data import Document
import numpy as np
import matplotlib.pyplot as plt
import pprint

 
if __name__ == '__main__':
    doc = Document("/home/salmon/workspace/SpDev/SpDB/examples/data/g086166.02990", format_type="GEQdsk")
    pprint.pprint(doc.root.holder)
