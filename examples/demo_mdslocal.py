import os
import pathlib
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np

from spdm import open_entry

os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"

# os.environ["t2_path"] = "/home/salmon/workspace/fytok_data/mdsplus/east/t2"

if __name__ == '__main__':

    entry = open_entry("file+mdsplus[EAST]:///home/salmon/workspace/fytok_data/mdsplus/~t/?tree_name=east[t1,t2,t3,t4,t5,t6]#70754")
    
    tf_current = entry.get(["tf", "coil", 0, "current", "data"])
    
    tf_time =entry.get(["tf", "coil", 0, "current", "time"])

    pprint({k:v for k,v in os.environ.items()  if k.endswith("_path")})
    
    pprint(tf_current)

    pprint(tf_time)

    
