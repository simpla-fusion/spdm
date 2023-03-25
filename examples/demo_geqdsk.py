import pprint
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from spdm.util.logger import logger
from spdm import open_entry
from spdm.data import File
from fytok.transport.Equilibrium import Equilibrium


if __name__ == '__main__':

    entry = open_entry("file+geqdsk[EAST]:///<Data path>/g080307.63000")

    logger.debug(entry.dump())
