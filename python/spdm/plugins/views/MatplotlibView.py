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

        self.draw(*objs, axis=axis, **kwargs)

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

    def draw(self, *obj: GeoObject,  axis, **kwargs):
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
        if isinstance(obj, Polygon):
            axis.add_patch(plt.Polygon(obj._points.transpose([1, 0]), **{"fill": False, "closed": True}))


__SP_EXPORT__ = MatplotlibView

# def plot(self, axis=None, *args, **kwargs):

#         if axis is None:
#             axis = plt.gca()

#         desc2d = self.description_2d[0]

#         # outline = desc2d.vessel.unit[0].annular.outline_inner

#         vessel_inner_points = np.array([desc2d.vessel.unit[0].annular.outline_inner.r,
#                                         desc2d.vessel.unit[0].annular.outline_inner.z]).transpose([1, 0])

#         vessel_outer_points = np.array([desc2d.vessel.unit[0].annular.outline_outer.r,
#                                         desc2d.vessel.unit[0].annular.outline_outer.z]).transpose([1, 0])

#         limiter_points = np.array([desc2d.limiter.unit[0].outline.r,
#                                    desc2d.limiter.unit[0].outline.z]).transpose([1, 0])

#         axis.add_patch(plt.Polygon(limiter_points, **
#                                    collections.ChainMap(kwargs.get("limiter", {}), {"fill": False, "closed": True})))

#         axis.add_patch(plt.Polygon(vessel_outer_points, **collections.ChainMap(kwargs.get("vessel_outer", {}),
#                                                                                kwargs.get("vessel", {}),
#                                                                                {"fill": False, "closed": True})))

#         axis.add_patch(plt.Polygon(vessel_inner_points, **collections.ChainMap(kwargs.get("vessel_inner", {}),
#                                                                                kwargs.get("vessel", {}),
#                                                                                {"fill": False, "closed": True})))

#         return axis
#    def plot(self, axis=None, *args, with_circuit=False, **kwargs):

#         if axis is None:
#             axis = plt.gca()

#         for coil in self.coil:
#             rect = coil.element[0].geometry.rectangle

#             axis.add_patch(plt.Rectangle((rect.r - rect.width / 2.0,  rect.z - rect.height / 2.0),
#                                          rect.width,  rect.height,
#                                          **collections.ChainMap(kwargs,  {"fill": False})))
#             axis.text(rect.r, rect.z, coil.name,
#                       horizontalalignment='center',
#                       verticalalignment='center',
#                       fontsize='xx-small')

#         return axis

# def plot(self, axis=None, *args, with_circuit=False, **kwargs):

#     if axis is None:
#         axis = plt.gca()
#     for idx, p_probe in enumerate(self.b_field_tor_probe):
#         pos = p_probe.position

#         axis.add_patch(plt.Circle((pos.r, pos.z), 0.01))
#         axis.text(pos.r, pos.z, idx,
#                   horizontalalignment='center',
#                   verticalalignment='center',
#                   fontsize='xx-small')

#     for p in self.flux_loop:
#         axis.add_patch(plt.Rectangle((p.position[0].r,  p.position[0].z), 0.01, 0.01))
#         axis.text(p.position[0].r, p.position[0].z, p.name,
#                   horizontalalignment='center',
#                   verticalalignment='center',
#                   fontsize='xx-small')
#     return axis
