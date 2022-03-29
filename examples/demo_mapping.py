import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm.logger import logger
from spdm.data.File import File
from spdm.data.Mapping import Mapping

if __name__ == '__main__':

    mapping = Mapping(mapping_path="/home/salmon/workspace/fytokdata/mapping")

    entry = mapping.find("EAST")

    logger.debug(entry.get("wall.description_2d.vessel.annular.outline_outer.r"))

    logger.debug(entry.get("wall.description_2d.vessel.annular.outline_outer.z"))

    m_entry = mapping.map(File("/home/salmon/public_data/efit_east", format="mdsplus").read(), source_schema="EAST")
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

    fg = plt.figure()

    plt.gca().add_patch(plt.Polygon(np.array([entry.get("wall.description_2d.vessel.annular.outline_outer.r"),
                                              entry.get("wall.description_2d.vessel.annular.outline_outer.z")]).transpose([1, 0]),
                                    fill=False, closed=True))

    # for coil in entry.pf_active.coil:
    #     rect = coil.element[0].geometry.rectangle.__value__()
    #     plt.gca().add_patch(plt.Rectangle((rect.r-rect.width/2.0, rect.z -
    #                                        rect.height/2.0), rect.width, rect.height, fill=False))

    logger.debug(entry.get(["equilibrium.time_slice", 0, "profiles_2d.psi"]))

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
