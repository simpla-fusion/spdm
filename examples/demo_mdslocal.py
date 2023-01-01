import os
import pathlib
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np

from spdm import open_entry

os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"


if __name__ == '__main__':

    method = 3
    
    if method == 1:
        entry = open_entry(
            "file+mdsplus[EAST]:///home/salmon/workspace/fytok_data/mdsplus/~t/?tree_name=east,t1,t2,t3,t4,t5,t6#70754")
    elif method == 2:
        entry = open_entry("ssh+mdsplus[EAST]://s108/share/arch/east/~t/~f~e~d/?tree_name=east,t1,t2,t3,t4,t5,t6#70754")
    else:
        os.environ['east_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t1_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t2_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t3_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t4_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t5_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t6_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'

        entry = open_entry("mdsplus[EAST]://#70754")

    tf_current = entry.get(["tf", "coil", 0, "current", "data"])

    tf_time = entry.get(["tf", "coil", 0, "current", "time"])

    pprint({k: v for k, v in os.environ.items() if k.endswith("_path")})

    pprint(tf_current)

    pprint(tf_time)

    pprint("DONE")
