import sys
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")



import numpy as np
import matplotlib.pyplot as plt
from spdm.util.logger import logger
from spdm.data import connect

if __name__ == '__main__':
    db = connect("imas://",
                 backend="mdsplus:///home/salmon/public_data/efit_east",  # "mdsplus://202.127.22.24/east_fit",
                 mapping=[
                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/static/config.xml",
                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/dynamic/config.xml"
                 ])

    entry = db.open(shot=55555).entry

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
        plt.gca().add_patch(plt.Rectangle((rect.r-rect.width/2.0, rect.z-rect.height/2.0), rect.width, rect.height, fill=False))

    for psi in entry.equilibrium.time_slice[0:10].profiles_2d[0].psi:
        logger.debug(psi.__value__())
        # plt.contour(
        #     time_slice.profiles_2d[0].grid.dim1.__value__(),
        #     time_slice.profiles_2d[0].grid.dim2.__value__(),
        #     time_slice.profiles_2d[0].psi.__value__(),
        #     levels=30, linewidths=0.4
        # )

    plt.axis('scaled')
    plt.savefig("imas_east.png")

    # for time_slice in entry.equilibrium.time_slice[:].boundary:
    #     logger.debug(time_slice.boundary.type.__value__())
