import collections.abc
import typing
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.Expression import Expression
from spdm.data.Field import Field
from spdm.data.Function import Function
from spdm.geometry.BBox import BBox
from spdm.geometry.Circle import Circle
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Line import Line
from spdm.geometry.Point import Point
from spdm.geometry.PointSet import PointSet
from spdm.geometry.Polygon import Polygon, Rectangle
from spdm.geometry.Polyline import Polyline
from spdm.utils.envs import SP_DEBUG
from spdm.utils.logger import SP_DEBUG, logger
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.utils.typing import array_type, as_array, is_array
from spdm.view.View import View


@View.register(["matplotlib"])
class MatplotlibView(View):
    backend = "matplotlib"

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _figure_post(self, fig: plt.Figure, title="", output=None, styles={}, transparent=True, **kwargs) -> typing.Any:
        fontsize = styles.get("fontsize", 16)

        fig.suptitle(title, fontsize=fontsize)

        fig.align_ylabels()

        fig.tight_layout()

        pos = fig.gca().get_position()
        height = pos.ymin+pos.ymax
        width = pos.xmax

        # width = fig.get_figwidth()
        # height = fig.get_figheight()

        fig.text(
            width + 0.01,
            0.5 * height,
            self.signature,
            verticalalignment="center",
            horizontalalignment="left",
            fontsize="small",
            alpha=0.2,
            rotation="vertical",
        )

        if output == "svg":
            buf = BytesIO()
            fig.savefig(buf, format="svg", transparent=transparent, **kwargs)
            buf.seek(0)
            fig_html = buf.getvalue().decode("utf-8")
            plt.close(fig)
            fig = fig_html

        elif output is not None:
            logger.debug(f"Write figure to  {output}")
            fig.savefig(output, transparent=transparent, **kwargs)
            plt.close(fig)
            fig = None

        return fig

    def render(self, obj, styles=None, view_point="rz", title=None, **kwargs) -> typing.Any:
        # if obj is None:
        #     return None
        
        if styles is None:
            styles = {}

        fig, canvas = plt.subplots()

        self._draw(canvas, obj, styles or {}, view_point=view_point)

        xlabel = styles.get("xlabel", None)

        if xlabel is not None:
            canvas.set_xlabel(xlabel)
        elif view_point.lower() == "rz":
            canvas.set_xlabel(r" $R$ [m]")
        else:
            canvas.set_xlabel(r" $X$ [m]")

        ylabel = styles.get("ylabel", None)
        if ylabel is not None:
            canvas.set_ylabel(ylabel)
        elif view_point.lower() == "rz":
            canvas.set_ylabel(r" $Z$ [m]")
        else:
            canvas.set_ylabel(r" $Y$ [m]")

        canvas.set_aspect("equal")
        canvas.axis("scaled")

        title = title or styles.get("title", None)

        return self._figure_post(fig, title=title, styles=styles, **kwargs)

    def _draw(self, canvas, obj: GeoObject | str | BBox, styles={}, view_point=None, **kwargs):
        if styles is False:
            return
        elif styles is None or styles is True:
            styles = {}

        s_styles = styles.get(f"${self.backend}", {})

        if obj is None or obj is _not_found_:
            pass

        elif isinstance(obj, tuple):
            o, s = obj
            if s is False:
                styles = False
            elif s is True:
                pass
            elif isinstance(s, collections.abc.Mapping):
                styles = merge_tree_recursive(styles, s)
            else:
                logger.warning(f"ignore unsupported styles {s}")

            self._draw(canvas, o, styles, view_point=view_point,   **kwargs)

        elif hasattr(obj.__class__, "__geometry__"):
            try:
                geo, s = obj.__geometry__(view_point=view_point, **kwargs)
                styles = merge_tree_recursive(styles, s)
            except Exception as error:
                if SP_DEBUG=="strict":
                    raise RuntimeError(f"ignore unsupported geometry {obj.__class__.__name__} {obj}! ") from error
            else:
                self._draw(canvas, geo, styles, view_point=view_point, **kwargs)

        elif isinstance(obj, dict):
            for k, o in obj.items():
                s = styles.get(k, {})
                if s is False:
                    continue

                self._draw(canvas, o, collections.ChainMap({"id": k}, s),
                           view_point=view_point, **kwargs)

            self._draw(canvas, None, styles, **kwargs)

        elif isinstance(obj, list):
            for idx, o in enumerate(obj):
                self._draw(canvas, o, collections.ChainMap({"id": idx}, styles), view_point=view_point, **kwargs)

            self._draw(canvas, None, styles, view_point=view_point,  **kwargs)

        elif isinstance(obj, (str, int, float, bool)):
            pos = s_styles.get("position", None)

            if pos is None:
                return

            canvas.text(
                *pos,
                str(obj),
                **collections.ChainMap(
                    s_styles,
                    {
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                        "fontsize": "xx-small",
                    },
                ),
            )

        elif isinstance(obj, BBox):
            canvas.add_patch(
                plt.Rectangle(obj.origin, *obj.dimensions, fill=False, **s_styles)
            )

        elif isinstance(obj, Polygon):
            canvas.add_patch(plt.Polygon(obj._points, fill=False, **s_styles))

        elif isinstance(obj, Polyline):
            canvas.add_patch(
                plt.Polygon(obj._points, fill=False, closed=obj.is_closed, **s_styles)
            )

        elif isinstance(obj, Line):
            canvas.add_artist(
                plt.Line2D([obj.p0.x, obj.p1.x], [obj.p0.y, obj.p1.y], **s_styles)
            )

        elif isinstance(obj, Curve):
            canvas.add_patch(
                plt.Polygon(obj._points, fill=False, closed=obj.is_closed, **s_styles)
            )

        elif isinstance(obj, Rectangle):
            canvas.add_patch(
                plt.Rectangle(
                    (obj._x, obj._y), obj._width, obj._height, fill=False, **s_styles
                )
            )

        elif isinstance(obj, Circle):
            canvas.add_patch(plt.Circle((obj.x, obj.y), obj.r, fill=False, **s_styles))

        elif isinstance(obj, Point):
            canvas.scatter(obj.x, obj.y, **s_styles)

        elif isinstance(obj, PointSet):
            canvas.scatter(*obj.points, **s_styles)

        elif isinstance(obj, GeoObject):
            self._draw(canvas, obj.bbox, styles)

        elif isinstance(obj, Field):
            R, Z = obj.mesh.points
            value = obj.__array__()

            levels = styles.pop("levels", s_styles.pop("levels", 10))

            canvas.contour(
                R,
                Z,
                value,
                levels=levels,
                **collections.ChainMap(s_styles, {"linewidths": 0.5}),
            )

        else:
            raise RuntimeError(f"Unsupport type {type(obj)} {obj}")

        text_styles = styles.get("text", False)
        if text_styles:
            if not isinstance(text_styles, dict):
                text_styles = {}

            if isinstance(obj, Line):
                text = obj.name
                pos = [obj.p1.x, obj.p1.y]
            elif isinstance(obj, GeoObject):
                text = obj.name
                pos = obj.bbox.center
            elif hasattr(obj, "mesh"):
                text = obj.name
                pos = obj.mesh.bbox.center
            else:
                text = str(obj)
                pos = None

            text_styles.setdefault("position", pos)

            self._draw(canvas, text, {f"${self.backend}": text_styles})

    def plot(self, obj, x_value: Expression | np.ndarray = None, x_label: str = None, x_axis: np.ndarray = None, stop_if_fail=False, **kwargs) -> typing.Any:

        styles = merge_tree_recursive(kwargs.pop("styles", {}), kwargs)

        fontsize = styles.get("fontsize", 16)

        if not isinstance(obj, list):  # draw as profiles
            obj = [obj]

        nprofiles = len(obj)

        fig, canvas = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2 * nprofiles))

        if nprofiles == 1:
            canvas = [canvas]

        if isinstance(x_value, np.ndarray):
            if x_axis is None:
                x_axis = x_value
            elif isinstance(x_axis, np.ndarray) and x_axis.size == x_value.size:
                pass
            else:
                raise TypeError(f"Unsupported x_value {x_value} {x_axis}")

        elif isinstance(x_value, Expression) and isinstance(x_axis, np.ndarray):
            if x_label is None:
                x_label = x_value.__label__

            x_value = x_value(x_axis)

        elif x_value is None:
            x_value = x_axis

        else:
            raise TypeError(f"Unsupported x_value {x_value} {x_axis}")

        for idx, profiles in enumerate(obj):
            if isinstance(profiles, tuple):
                profiles, sub_styles = profiles
            else:
                sub_styles = {}

            if sub_styles is False:
                continue

            y_label = sub_styles.get("y_label", None) or getattr(profiles, "__label__", "")

            if not isinstance(profiles, (list)):
                profiles = [profiles]

            for p in profiles:
                try:
                    self._plot(canvas[idx], p, x_value=x_value, x_axis=x_axis,
                               styles=collections.ChainMap(sub_styles, styles))
                except Exception as error:
                    # if stop_if_fail:
                    raise RuntimeError(f"Plot [index={idx}] failed! y_label= \"{y_label}\"  ") from error
                    # else:
                    #     logger.debug(f'Plot [index={idx}] failed! y_label= "{y_label}"  [{error}] ')

            canvas[idx].legend(fontsize=fontsize)
            canvas[idx].set_ylabel(ylabel=y_label, fontsize=fontsize)

        canvas[-1].set_xlabel(x_label, fontsize=fontsize)

        return self._figure_post(fig, styles=styles, **kwargs)

    def _plot(self, canvas, obj, x_axis: array_type, x_value: array_type,   **kwargs):
        if obj is None or obj is _not_found_:
            return
        # elif not isinstance(x_value, array_type) and not isinstance(x_axis, array_type):
        #     raise RuntimeError(f"Unsupported x_value {x_value} {x_axis} ")

        if isinstance(obj, tuple):
            obj, styles = obj
        else:
            styles = {}

        styles = merge_tree_recursive(styles, kwargs.pop("styles", {}), kwargs)

        s_styles = styles.get(f"${self.backend}", {})

        label = styles.get("label", None) or kwargs.get("label", None)

        data = []
        if isinstance(obj, Function) and x_value is None:
            x_value = obj.dims[0]
            y_value = obj.__array__()
            label = label or obj.__label__
            data = [x_value, y_value]

        elif isinstance(obj, Expression):
            label = label or getattr(obj, "name", None) or getattr(obj, "__label__", None) or str(obj)
            y_value = obj(x_axis)
            data = [x_value, y_value]

        elif is_array(obj):
            y_value = obj
            label = " "
            if x_value is not None and x_value.size == obj.size:
                data = [x_value, y_value]
            else:
                data = [y_value]

        elif np.isscalar(obj):
            y_value = np.full_like(x_value, obj, dtype=float)
            data = [x_value, y_value]

        else:
            logger.warning(f"ignore unsupported profiles label={label}")
            return
            # raise RuntimeError(f"Unsupported profiles {obj}")

        canvas.plot(*data, **s_styles, label=label)

    # def profiles_(self, obj, *args,  x_axis=None, x=None,
    #               default_num_of_points=128, fontsize=10, grid=True,
    #               signature=None, title=None, **kwargs):
    #     fontsize = kwargs.get("fontsize", 10)

    #     nprofiles = len(obj)

    #     fig, canves = plt.subplots(
    #         ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2 * nprofiles)
    #     )

    #     self.draw(canves, obj, styles)

    #     x_label = kwargs.get("xlabel", "")

    #     if len(canves) == 1:
    #         canves[0].set_xlabel(x_label, fontsize=fontsize)
    #     else:
    #         canves[-1].set_xlabel(x_label, fontsize=fontsize)

    #     if not isinstance(profile_list, collections.abc.Sequence):
    #         profile_list = [profile_list]

    #     if isinstance(x_axis, collections.abc.Sequence) and not isinstance(
    #         x_axis, np.ndarray
    #     ):
    #         x_axis, x_label, *x_opts = x_axis
    #         x_opts = (x_opts or [{}])[0]
    #     else:
    #         x_axis = [0, 1]
    #         x_label = ""
    #         x_opts = {}

    #     if isinstance(x_axis, Function) and x is not None:
    #         x_axis = x_axis(x)
    #     elif x is None and isinstance(x_axis, np.ndarray):
    #         x = x_axis

    #     if isinstance(x_axis, np.ndarray):
    #         x_min = x_axis[0]
    #         x_max = x_axis[-1]
    #     elif isinstance(x_axis, collections.abc.Sequence) and len(x_axis) == 2:
    #         x_min, x_max = x_axis
    #         x_axis = np.linspace(x_min, x_max, default_num_of_points)
    #     else:
    #         raise TypeError(x_axis)

    #     if x is None and isinstance(x_axis, np.ndarray):
    #         x = x_axis
    #     elif callable(x_axis) or isinstance(x_axis, Function):
    #         x_axis = x_axis(x)

    #     nprofiles = len(profile_list)

    #     fig, sub_plot = plt.subplots(
    #         ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2 * nprofiles)
    #     )

    #     if not isinstance(sub_plot, (collections.abc.Sequence, np.ndarray)):
    #         sub_plot = [sub_plot]

    #     for idx, profile_grp in enumerate(profile_list):
    #         if not isinstance(profile_grp, list):
    #             profile_grp = [profile_grp]
    #         ylabel = None
    #         for jdx, p_desc in enumerate(profile_grp):
    #             profile, label, *o_args = p_desc
    #             opts = {}
    #             if len(o_args) > 0 and ylabel is None:
    #                 ylabel = o_args[0]
    #             if len(o_args) > 1:
    #                 opts = o_args[1]

    #             y = None

    #             if isinstance(profile, Function) or callable(profile):
    #                 try:
    #                     y = profile(x)
    #                 except Exception as error:
    #                     raise RuntimeError(
    #                         f"Can not get profile [idx={idx} jdx={jdx}]! name={getattr(profile,'_name',profile)}\n {error} "
    #                     ) from error

    #             elif isinstance(profile, np.ndarray) and len(profile) == len(x):
    #                 y = profile
    #             elif np.isscalar(profile):
    #                 y = np.full_like(x, profile, dtype=float)
    #             else:
    #                 raise RuntimeError(f"Illegal profile! {profile}!={x}")

    #             if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
    #                 logger.warning(f"Illegal profile! {(type(x) ,type(y), label, o_args)}")
    #                 continue
    #             elif x.shape != y.shape:
    #                 logger.warning(f"Illegal profile! {x.shape} !={y.shape}")
    #                 continue
    #             else:
    #                 # 删除 y 中的 nan
    #                 mark = np.isnan(y)
    #                 # if np.any(mark):
    #                 #     logger.warning(f"Found NaN in array  {np.argwhere(mark)}! {profile}  ")
    #                 sub_plot[idx].plot(x_axis[~mark], y[~mark], label=label, **opts)

    #         sub_plot[idx].legend(fontsize=fontsize)

    #         if grid:
    #             sub_plot[idx].grid()

    #         if ylabel is not None:
    #             sub_plot[idx].set_ylabel(ylabel, fontsize=fontsize)
    #         sub_plot[idx].labelsize = "media"
    #         sub_plot[idx].tick_params(labelsize=fontsize)

    #     if len(sub_plot) <= 1:
    #         sub_plot[0].set_xlabel(x_label, fontsize=fontsize)

    #     else:
    #         sub_plot[-1].set_xlabel(x_label, fontsize=fontsize)

    #     return fig

    # def draw_profile(self, profiles, x_axis, canves: plt.Axes = ..., style=None, **kwargs):
    #     if style is None:
    #         style = {}

    #     fontsize = style.get("fontsize", 10)

    #     ylabel = None

    #     x_value = x_axis

    #     if not isinstance(profiles, collections.abc.Sequence):
    #         profiles = [profiles]

    #     for profile, label, legend, *opts in profiles:
    #         y = None

    #         if isinstance(profile, Function) or callable(profile):
    #             try:
    #                 y = profile(x_value)
    #             except Exception as error:
    #                 raise RuntimeError(
    #                     f"Can not get profile! name={getattr(profile,'name',profile)}\n {error} "
    #                 ) from error

    #         elif isinstance(profile, array_type) and len(profile) == len(x_value):
    #             y = profile
    #         elif np.isscalar(profile):
    #             y = np.full_like(x_value, profile, dtype=float)
    #         else:
    #             raise RuntimeError(f"Illegal profile! {profile}!={x_value}")

    #         if not isinstance(y, array_type) or not isinstance(x_value, array_type):
    #             logger.warning(
    #                 f"Illegal profile! {(type(x_value) ,type(y), label, opts)}"
    #             )
    #             continue
    #         elif x.shape != y.shape:
    #             logger.warning(f"Illegal profile! {x_value.shape} !={y.shape}")
    #             continue
    #         else:
    #             # 删除 y 中的 nan
    #             mark = np.isnan(y)
    #             # if np.any(mark):
    #             #     logger.warning(f"Found NaN in array  {np.argwhere(mark)}! {profile}  ")
    #             canves.plot(x_axis[~mark], y[~mark], label=label, **opts)

    #     canves.legend(fontsize=fontsize)

    #     if kwargs.get("grid", True):
    #         canves.grid()

    #     if ylabel is not None:
    #         canves.set_ylabel(ylabel, fontsize=fontsize)
    #     canves.labelsize = "media"
    #     canves.tick_params(labelsize=fontsize)


# def sp_figure_signature(fig: plt.Figure, signature=None, x=1.0, y=0.1):
#     if signature is False:
#         return fig
#     elif not isinstance(signature, str):
#         signature = f"author: {getpass.getuser().capitalize()}. Create by SpDM at {datetime.datetime.now().isoformat()}."

#     pos = fig.gca().get_position()

#     fig.text(pos.xmax+0.01, 0.5*(pos.ymin+pos.ymax), signature,
#              verticalalignment='center', horizontalalignment='left',
#              fontsize='small', alpha=0.2, rotation='vertical')

#     # fig.text(x, y, signature, va='bottom', ha='left', fontsize='small', alpha=0.5, rotation='vertical')
#     return fig

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
