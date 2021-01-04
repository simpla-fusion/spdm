import collections
import functools
import os
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.DataObject import DataObject
from spdm.data.File import File
from spdm.flow.ModuleRepository import ModuleRepository
from spdm.flow.SpModule import SpModule
from spdm.util.logger import logger

if __name__ == "__main__":

    os.environ["FUYUN_CONFIGURE_PATH"] = "/home/salmon/workspace/SpDev/SpDB/examples/data/FuYun/configure.yaml"

    module = ModuleRepository(repo_name='FuYun', repo_tag='FY')

    module.factory.insert_handler("SpModule", SpModule)

    Genray = module.new_class("physics/genray", version="10.13_200117", tag="-gompi-2020a")

    cfg = {
        "tokamak": {
            "eqdskin":  {"$class": "file.geqdsk", "path": "{FY_MODULEFILE_DIR}/../template/g063982.04800"}},
        "genr": {
            "partner":  {"$class": "file.netcdf", "path": "/home/salmon/workspace/data/genray/genray_profs_in.nc"},
            "outdat": "{WORKING_DIR}"
        },
        "ecocone": {"gzone": 1}
    }

    genray = Genray(num_of_steps=1, dt=0.001, config=cfg, working_dir="/home/salmon/workspace/output")

    logger.debug(genray.outputs.EXITCODE)

    # out_nc = genray_out.out_nc
    # out_eq = genray_out.out_eq

    # wr = out_nc.entry.wr[1]/100.0
    # wz = out_nc.entry.wz[:]/100.0

    # plt.plot(out_eq.limrz[:, 0], out_eq.limrz[:, 1])
    # plt.plot(out_eq.bbsrz[:, 0], out_eq.bbsrz[:, 1])

    # plt.plot(wr, wz)
    # plt.contour(
    #     out_nc.eqdsk_r[:, :],
    #     out_nc.eqdsk_z[:, :],
    #     out_nc.eqdsk_psi[:, :])

    # plt.savefig("../output/demo_module.svg")

    logger.debug("Done")
