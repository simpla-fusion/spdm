import pprint
import subprocess
import pathlib
import numpy as np

from spdm.data import io
from ..util.logger import logger

if __name__ == "__main__":

    # subprocess.run(['mongod',
    #                 '--dbpath', pathlib.Path.cwd(),
    #                 '--logpath', pathlib.Path.cwd()/"mongodb.log",
    #                 '--bind_ip', '127.0.0.1'
    #                 ])

    c1 = io.connect("mongodb://127.0.0.1:27017/EAST/diag/imas")
    # collection.insert_one({"a": "Just a test", "b": 5})
    # c1 = io.connect("mdsplus://127.0.0.1/EAST")

    for i in range(10):
        c1.insert_one(
            {"device": {"name": "EAST", "shot": 555}, "num": i,
             "data": {
                 "x": np.random.rand(5, 6),
                "y": np.random.rand(5, 6),
            }
            })

    pprint.pprint([d["data"] for d in c1.find()])
    # pprint.pprint([d for d in c1.find(projection="data")])
