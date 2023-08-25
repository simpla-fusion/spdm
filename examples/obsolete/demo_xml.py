import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


import matplotlib.pyplot as plt
import numpy as np
from spdm.data import Document
from ..util.logger import logger



# import os

# os.environ['east_path'] = 'mds.ipp.ac.cn::/pcs_east'


if __name__ == '__main__':
    entry = Document([
        "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/static/config.xml",
        "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/dynamic/config.xml"
    ], format_type="XML").entry

    fg = plt.figure()

    plt.gca().add_patch(plt.Polygon(np.array([entry.wall.description_2d.vessel.annular.outline_outer.r.__value__(),
                                              entry.wall.description_2d.vessel.annular.outline_outer.z.__value__()]).transpose([1, 0]),
                                    fill=False, closed=True))

    plt.gca().add_patch(plt.Polygon(np.array([entry.wall.description_2d.vessel.annular.outline_inner.r.__value__(),
                                              entry.wall.description_2d.vessel.annular.outline_inner.z.__value__()]).transpose([1, 0]),
                                    fill=False, closed=True))

    plt.gca().add_patch(plt.Polygon(np.array([entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                                              entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]).transpose([1, 0]),
                                    fill=False, closed=True))

    for coil in entry.pf_active.coil:
        rect = coil.element[0].geometry.rectangle.__value__()
        logger.debug(rect)
        plt.gca().add_patch(plt.Rectangle((rect.r-rect.width/2.0, rect.z-rect.height/2.0), rect.width, rect.height, fill=False))

    # logger.debug([time_slice.profiles_2d[0].boundary.type.__value__()
    #               for time_slice in entry.equilibrium.time_slice[1:10]])

    # plt.contour(
    #     entry.equilibrium.time_slice[1].profiles_2d.Mesh.dim1.__value__(),
    #     entry.equilibrium.time_slice[1].profiles_2d.Mesh.dim2.__value__(),
    #     entry.equilibrium.time_slice[1].profiles_2d.psi.__value__(),
    #     levels =30,linewidths=0.4
    #     )
    plt.axis('scaled')

    # for time_slice in entry.equilibrium.time_slice[:]:
    #     logger.debug(time_slice.ids_properties.homogeneous_time.__value__())
    plt.savefig("imas_east.png")
