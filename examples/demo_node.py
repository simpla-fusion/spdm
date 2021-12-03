from copy import copy, deepcopy

import numpy as np
from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.List import List, _next_
from spdm.data.Node import Node
from spdm.data.Path import Path


if __name__ == '__main__':
    cache = {
        "a": [
            "hello world {name}!",
            "hello world2 {name}!",
            1.0, 2, 3, 4
        ],
        "c": "I'm {age}!",
        "d": {
            "e": "{name} is {age}",
            "f": "{address}"
        }
    }
    d = Dict(cache)

    print(d["d"]["e"])

    d["b"] = "hello world {name}!"

    d["h"].append(1234)
    d["h"].append(5678)

    d.update({"d": {"g": 5}})

    print(cache)
