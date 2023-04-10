import os
from pprint import pprint

from spdm.data.open_entry import open_entry

os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"


if __name__ == '__main__':

    method = 2

    if method == 1:
        entry = open_entry(
            "file+MDSplus[EAST]:///home/salmon/workspace/fytok_data/mdsplus/~t/?tree_name=east,t1,t2,t3,t4,t5,t6#70754")
    elif method == 2:
        entry = open_entry("ssh+MDSplus[EAST]://s108/share/arch/east/~t/~f~e~d/?tree_name=east,t1,t2,t3,t4,t5,t6#70754")
    else:
        os.environ['east_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t1_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t2_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t3_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t4_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t5_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'
        os.environ['t6_path'] = '/home/salmon/workspace/fytok_data/mdsplus/~t'

        entry = open_entry("MDSplus[EAST]://#70754")

    tf_current = entry.child("tf/coil/0/current/data").query()

    tf_time = entry.child("tf/coil/0/current/time").query()

    # pprint({k: v for k, v in os.environ.items() if k.endswith("_path")})

    pprint(tf_current)

    pprint(tf_time)

    pprint("DONE")
