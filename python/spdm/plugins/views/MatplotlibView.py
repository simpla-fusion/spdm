import collections.abc
import typing
from io import BytesIO

import matplotlib.pyplot as plt
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Polygon
from spdm.utils.logger import logger
from spdm.views.View import View


@View.register(["matplotlib", "Matplotlib"])
class MatplotlibView(View):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def display(self, *objs, **kwargs) -> typing.Any:

        fig, axis = plt.subplots()

        if len(objs) == 1 and isinstance(objs[0], collections.abc.Sequence):
            objs = objs[0]

        for obj in objs:
            if isinstance(obj, tuple):
                obj, opts = obj
            else:
                opts = {}

            if hasattr(obj, "plot"):
                axis = obj.plot(axis, **opts)
            else:
                logger.error(f"Can not plot {obj}")

        axis.set_aspect('equal')
        axis.axis('scaled')
        axis.set_xlabel(kwargs.pop("xlabel", ""))
        axis.set_ylabel(kwargs.pop("ylabel", ""))

        fig.suptitle(kwargs.pop("title", ""))
        fig.align_ylabels()
        fig.tight_layout()

        pos = axis.get_position()

        fig.text(pos.xmax+0.01, 0.5*(pos.ymin+pos.ymax), self.signature,
                 verticalalignment='center', horizontalalignment='left',
                 fontsize='small', alpha=0.2, rotation='vertical')

        res = "<Nothing to display />"

        schema = kwargs.pop("schema", self.schema)

        if schema is "html":
            buf = BytesIO()
            fig.savefig(buf, format='svg', transparent=True)
            buf.seek(0)
            res = buf.getvalue().decode('utf-8')
            plt.close(fig)
            return res
        else:
            if "output" in kwargs:
                fig.savefig(kwargs.pop("output"), transparent=True)

            return fig

    def draw(self, axis, obj: GeoObject, **kwargs):

        if isinstance(obj, Polygon):
            axis.add_patch(plt.Polygon(obj._points.transpose([1, 0]), **{"fill": False, "closed": True}))


__SP_EXPORT__ = MatplotlibView
