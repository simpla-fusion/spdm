from spdm.data import file
import pprint
from ..util.logger import logger

if __name__ == "__main__":
    d = File("/home/salmon/workspace/fytok/external/SpDB/examples/data/g063982.04800",
                schema="geqdsk")
    print(d._data)
    # io.save(d, "test.json")
    # io.save(d, "test.yaml")
    # # io.save(d, "test.nml")
    # db = io.connect("local://", schema="json")
    # # idx = db.save(d)
    # db.load(idx)
