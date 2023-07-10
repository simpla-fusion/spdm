from spdm.data.Expression import Expression, Variable
from spdm.data.ExprOp import ExprOp
from spdm.data.Function import Function
import numpy as np
from spdm.utils.logger import logger


if __name__ == '__main__':
    _x = Variable(0, "x")
    _y = Variable(1, "y")
    expr = (np.cos(_x)+np.sin(_y))*2
    logger.debug(expr._repr_latex_())
