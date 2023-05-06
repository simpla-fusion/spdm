
import numpy as np
from scipy import constants
from spdm.utils.logger import logger
from spdm.data.Function import Expression, Function


if __name__ == '__main__':
    x1 = np.linspace(0, 1, 128)
    x2 = np.linspace(0, 2, 128)
    fun1 = Function(np.sin(x1), x1)
    fun2 = Function(np.cos(x2), x2)

    expr = fun1 + fun2
    logger.debug(expr)

    logger.debug(expr(x2))
