import pprint

import numpy as np

from spdm.data import io
from spdm.util.logger import logger

if __name__ == "__main__":

    # c1 = io.connect("EAST/east_*.h5")
    # collection.insert_one({"a": "Just a test", "b": 5})
    c1 = io.connect("mdsplus://127.0.0.1/EAST")

    for i in range(10):
        c1.insert_one(
            {"device": {"name": "EAST", "shot": 555}, "num": i,
             "data": np.random.rand(5, 6)})

    pprint.pprint([d["data"] for d in c1.find(
        projection={"data": 1, "_id": 0})])
    pprint.pprint([d for d in c1.find(projection="data")])
