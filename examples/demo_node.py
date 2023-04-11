from copy import copy, deepcopy

import numpy as np
from spdm.util.logger import logger
from spdm.data.Dict import Dict
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property


class Foo(Dict[Node]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    foo: str = sp_property()  # type: ignore
    doo: int = sp_property(default_value=10)  # type: ignore

    @sp_property
    def goo(self) -> int:
        return self.doo+100


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
            "foo": "{address}"
        }
    }
    d = Dict[Foo](cache)
    logger.debug(d["d"])
    tmp = d["d"]
    logger.debug(tmp["foo"])
    logger.debug(tmp.foo)
    logger.debug(tmp["foo"])
    logger.debug(tmp.foo)
    logger.debug(tmp["doo"])
    logger.debug(tmp.doo)
    logger.debug(tmp["doo"])
    logger.debug(tmp.goo)
    logger.debug(tmp["goo"])
    # d["b"] = "hello world {name}!"

    # d["h"].append(1234)
    # d["h"].append(5678)

    # d.update({"d": {"g": 5}})

    # print(cache)
