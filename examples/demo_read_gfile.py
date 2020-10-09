from spdm.data import file
import pprint
from spdm.util.logger import logger

if __name__ == "__main__":
    d = file.load("/home/salmon/workspace/spdm/examples/python/east_04010.gfile",
                schema="geqdsk")
    print(d._data)
    # io.save(d, "test.json")
    # io.save(d, "test.yaml")
    # # io.save(d, "test.nml")
    # db = io.connect("local://", schema="json")
    # # idx = db.save(d)
    # db.load(idx)
