from spdm.data.Expression import Expression, Variable
from spdm.data.Functor import Functor
from spdm.numlib.calculus import derivative
from spdm.data.Function import Function
import numpy as np
from spdm.utils.logger import logger


if __name__ == '__main__':
    _x = Variable(0, "x")
    _y = Variable(1, "y")
    expr = derivative(1, (np.cos(_x)+np.sin(_y))*2)
    logger.debug(expr._repr_latex_())
