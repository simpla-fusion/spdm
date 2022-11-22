import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm.logger import logger
from spdm.data.File import File
from spdm.data.Mapping import Mapping
from fytok.modules.transport.Equilibrium import Equilibrium
from fytok.modules.device.PFActive import PFActive
from fytok.modules.device.Wall import Wall


if __name__ == '__main__':

    entry = Mapping("/home/salmon/workspace/fytok_data/mapping")\
        .map(File("/home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300",
                  format="mdsplus").read(), source_schema="EAST")

    fig = plt.figure()
    axis = fig.gca()

    time_slice = 50

    desc = entry.get(["equilibrium", "time_slice", time_slice]).dump()

    desc["time"] = 1.2345

    desc["vacuum_toroidal_field"] = {
        "b0": entry.get(["equilibrium", "vacuum_toroidal_field", "b0"])[time_slice],
        "r0": entry.get(["equilibrium", "vacuum_toroidal_field", "r0"])[time_slice],
    }

    eq = Equilibrium(desc)

    logger.debug(eq.time)
    logger.debug(desc["global_quantities"]["ip"])

    eq.plot(axis, contour=np.linspace(0, 5, 50))

    pf_active = PFActive(entry.get(["pf_active"]))

    for coil in pf_active.coil:
        logger.debug(coil.element[0].geometry.rectangle)
        logger.debug(coil.current.data[100])

    pf_active.plot(axis)

    wall = Wall(entry.get(["wall"]))

    wall.plot(axis)

    axis.set_aspect('equal')
    axis.axis('scaled')
    axis.set_xlabel(r"Major radius $R$ [m]")
    axis.set_ylabel(r"Height $Z$ [m]")

    fig.savefig("/home/salmon/workspace/output/tokamak.png", transparent=True)
