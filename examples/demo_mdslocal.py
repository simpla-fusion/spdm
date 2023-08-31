import os

from spdm.utils.logger import logger
from spdm.data.open_entry import open_entry

os.environ["SP_DATA_MAPPING_PATH"]="/ssd01/salmon_work/workspace/fytok/python/fytok/_mapping"

if __name__ == '__main__':

    method = 1
    DATA_PATH="/ssd01/salmon_work/workspace/fytok_data"
    if method == 1:
        entry = open_entry(f"file+MDSplus[EAST]://{DATA_PATH}/mdsplus/~t/?tree_name=efit_east#70745")
    elif method == 2:
        entry = open_entry(f"file+MDSplus[EAST]:///share/arch/east/~t/~f~e~d/?tree_name=east,t1,t2,t3,t4,t5,t6#70754")
    else:
        os.environ['east_path'] = f'{DATA_PATH}/mdsplus/~t'
        os.environ['t1_path'] =   f'{DATA_PATH}/mdsplus/~t'
        os.environ['t2_path'] =   f'{DATA_PATH}/mdsplus/~t'
        os.environ['t3_path'] =   f'{DATA_PATH}/mdsplus/~t'
        os.environ['t4_path'] =   f'{DATA_PATH}/mdsplus/~t'
        os.environ['t5_path'] =   f'{DATA_PATH}/mdsplus/~t'
        os.environ['t6_path'] =   f'{DATA_PATH}/mdsplus/~t'

        entry = open_entry("MDSplus[EAST]:///#70754")

    pf_current = entry.child("pf_active/coil/0/current/data").fetch()

    pf_time = entry.child("pf_active/coil/0/current/time").fetch()

    # # pprint({k: v for k, v in os.environ.items() if k.endswith("_path")})

    logger.debug(pf_current)

    logger.debug(pf_time)

    # time_slice = 100

    # eq = entry.child(f"equilibrium/time_slice/{time_slice}").fetch()

    # logger.debug(eq)

    logger.debug("DONE")
