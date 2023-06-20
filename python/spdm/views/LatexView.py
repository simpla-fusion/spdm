import collections.abc
import typing
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Expression import Expression
from spdm.data.ExprOp import ExprOp
from spdm.data.Function import Function
from spdm.utils.logger import logger
from spdm.utils.typing import array_type
from spdm.views.View import View


@View.register(["Latex", "latex"])
class LatexView(View):
    backend = "latex"

    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def render(self, expr: Expression, *args, **kwargs) -> typing.Any:
        if isinstance(expr, np.ndarray):
            res = f"<{expr.shape}>"
        elif np.isscalar(expr):
            res = f"{expr}"
        elif isinstance(expr._op, ExprOp) and expr._op.name is not None:
            if len(expr._children) == 2 and expr._op.tag is not None:
                res = f"({self.render(expr._children[0])} {expr._op.tag} {self.render(expr._children[1])})"
            else:
                res = f"{expr._op.name}({', '.join([self.render(arg) for arg in expr._children])})"
        else:
            res = f"{expr.__name__}"
        return res
