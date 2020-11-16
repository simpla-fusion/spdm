
import sys
sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == '__main__':
    from spdm.util.logger import logger
    from spdm.data.Entry import open_entry

    entry = open_entry("east+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east#shot=55555", itime=100)

    # logger.debug(entry.pf_active.coil[1].element[0].geometry.rectangle.r())
    # logger.debug(entry.wall.description_2d[0].limiter.unit[0].outline.r())
    # logger.debug(entry.wall.description_2d[0].limiter.unit[0].outline.r())
    logger.debug(type(entry.equilibrium.time_slice[10].profiles_1d.psi()))
    logger.debug(entry.equilibrium.time_slice[10].profiles_1d.dpressure_dpsi())
