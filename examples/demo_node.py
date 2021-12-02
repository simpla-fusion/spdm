from copy import copy, deepcopy

import numpy as np
from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.List import List, _next_
from spdm.data.Node import Node
from spdm.data.Path import Path


if __name__ == '__main__':

    d = List(["a"])

    print(d[0])
