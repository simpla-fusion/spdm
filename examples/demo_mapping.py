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

    mapping = Mapping(mapping_path="/home/salmon/workspace/fytok_data/mapping")

    entry = mapping.map(File("/home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300",
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

    # logger.debug(eq.vacuum_toroidal_field.r0)
    # logger.debug(eq.profiles_1d.f(np.linspace(0, 1.0, 32)))

    # db = Collection("mapping://",
    #                 source="mdsplus:///home/salmon/public_data/efit_east",
    #                 id_hasher="{shot}",  #
    #                 mapping=[
    #                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/static/config.xml",
    #                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/dynamic/config.xml"
    #                 ])
    # db = Collection("EAST:///home/salmon/public_data/~t/",
    #                 default_tree_name="efit_east")

    # entry = db.open(shot=55555).entry

    # plt.gca().add_patch(plt.Polygon(np.array([entry_efit.get("wall.description_2d.vessel.annular.outline_outer.r"),
    #                                           entry_efit.get("wall.description_2d.vessel.annular.outline_outer.z")]).transpose([1, 0]),
    #                                 fill=False, closed=True))

    # for coil in entry_efit.get("pf_active.coil"):
    #     rect = coil.get(["element", 0, "geometry", "rectangle"]).dump_named()
    #     plt.gca().add_patch(plt.Rectangle((rect.r-rect.width/2.0, rect.z -
    #                                        rect.height/2.0), rect.width, rect.height, fill=False))
    # plt.axis('scaled')

    # # m_entry = mapping.map(File("/home/salmon/workspace/data/efit_east?tree_name=efit_east,shot=38300",
    # #                            format="mdsplus").read(), source_schema="EAST")
    # logger.debug(entry_efit.get(["equilibrium", "time_slice", 12]).dump())

    # logger.debug(entry_efit.get(["equilibrium", "time_slice", 2, "profiles_2d", "psi"]))

    # logger.debug(entry.get(["equilibrium.time_slice", 0, "profiles_2d.psi"]))

    # plt.contour(
    #     entry.equilibrium.time_slice[1].profiles_2d.grid.dim1.__value__(),
    #     entry.equilibrium.time_slice[1].profiles_2d.grid.dim2.__value__(),
    #     entry.equilibrium.time_slice[1].profiles_2d.psi.__value__(),
    #     levels=30, linewidths=0.4
    # )
    # for time_slice in entry.equilibrium.time_slice[0:10]:
    #     plt.contour(
    #         time_slice.profiles_2d[0].grid.dim1.__value__(),
    #         time_slice.profiles_2d[0].grid.dim2.__value__(),
    #         time_slice.profiles_2d[0].psi.__value__(),
    #         levels=30, linewidths=0.4
    #     )
    # plt.axis('scaled')
    # plt.savefig("imas_east.png")

    # # doc = Document("/home/salmon/workspace/output/east/test_a.json", mode="w")
    # # doc.copy(entry.wall)
    # # doc.save()
    # logger.debug("DONE")

    # for time_slice in entry.equilibrium.time_slice[:].boundary:
    #     logger.debug(time_slice.boundary.type.__value__())
