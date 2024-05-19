import collections.abc
import typing
import uuid

import bokeh
import bokeh.models.widgets as bk
import ipywidgets as ip
import jupyter_bokeh as jbk
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure
from spdm.core.HTree import HTree
from spdm.geometry.BBox import BBox
from spdm.geometry.Circle import Circle
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Line import Line
from spdm.geometry.Point import Point
from spdm.geometry.PointSet import PointSet
from spdm.utils.logger import logger

from spdm.view.View import View


def bokeh_example():
    output_notebook()
    x = np.linspace(0, 2*np.pi, 2000)
    y = np.sin(x)
    p = figure(title="simple line example", width=600, height=300, y_range=(-5, 5),
               background_fill_color='#efefef')
    r = p.line(x, y, color="#8888cc", line_width=1.5, alpha=0.8)

    def update(f, w=1, A=1, phi=0):
        if f == "sin":
            func = np.sin
        elif f == "cos":
            func = np.cos
        else:
            func = np.sin
        r.data_source.data['y'] = A * func(w * x + phi)
    jbk.BokehModel(p)
    ip.interact(update, f=["sin", "cos"], w=(0, 50), A=(1, 10), phi=(0, 20, 0.1))


@View.register(["bokeh", "Bokeh"])
class BokehView(View):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def draw(self, obj, **kwargs) -> typing.Any:
        """
            Bokeh + Jupyter 生成可交互的widget
        """

        return obj
