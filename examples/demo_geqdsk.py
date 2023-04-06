
import matplotlib.pyplot as plt
from spdm.util.logger import logger
from spdm.data.open_entry import open_entry
from fytok.transport.Equilibrium import Equilibrium


if __name__ == '__main__':

    # entry = open_entry("file+geqdsk[EAST]:///<Data path>/g080307.63000").pull()

    eq = Equilibrium({})
    # logger.debug(entry.dump())
