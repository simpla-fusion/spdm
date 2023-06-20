import collections.abc
import typing
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from spdm.data.Function import Function
from spdm.geometry.Circle import Circle
from spdm.geometry.GeoObject import GeoObject
from spdm.geometry.Polygon import Polygon, Rectangle
from spdm.geometry.Polyline import Polyline
from spdm.geometry.Point import Point
from spdm.geometry.Curve import Curve
from spdm.geometry.BBox import BBox
from spdm.utils.logger import logger
from spdm.utils.typing import array_type
from spdm.views.View import View


@View.register(["matplotlib", "Matplotlib"])
class MatplotlibView(View):
    backend = "matplotlib"

    def __init__(self, *args, view_point=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._view_point = view_point  # TODO: 未实现, for 3D view

    def render(self, obj, styles=None, **kwargs) -> typing.Any:

        fig, canves = plt.subplots()

        self.draw(canves, obj, styles)

        canves.set_aspect('equal')
        canves.axis('scaled')

        return self._render_post(fig, **kwargs)

    def _render_post(self, fig, **kwargs) -> typing.Any:

        fig.suptitle(kwargs.pop("title", ""))
        fig.align_ylabels()
        fig.tight_layout()

        pos = fig.gca().get_position()

        fig.text(pos.xmax+0.01, 0.5*(pos.ymin+pos.ymax), self.signature,
                 verticalalignment='center', horizontalalignment='left',
                 fontsize='small', alpha=0.2, rotation='vertical')

        output = kwargs.pop("output", None)

        if output == "svg":
            buf = BytesIO()
            fig.savefig(buf, format='svg', transparent=True)
            buf.seek(0)
            fig_html = buf.getvalue().decode('utf-8')
            plt.close(fig)
            fig = fig_html
        elif output is not None:
            fig.savefig(output, transparent=True)
            plt.close(fig)
            fig = None
        return fig

    def _draw(self, canvas, obj: typing.Any,  styles={}):
        if styles is False:
            return

        s_styles = styles.get(f"${self.backend}", {})

        if obj is None:
            pass
        elif isinstance(obj, (str, int, float, bool)):
            pos = s_styles.pop("position", None)

            if pos is None:
                return

            canvas.text(*pos, str(obj),
                        horizontalalignment=s_styles.pop('horizontalalignment', 'center'),
                        verticalalignment=s_styles.pop('verticalalignment', 'center'),
                        fontsize=s_styles.pop('fontsize', 'xx-small'),
                        ** s_styles
                        )

        elif isinstance(obj, BBox):
            canvas.add_patch(plt.Rectangle(obj.origin, *obj.dimensions, fill=False, **s_styles))

        elif isinstance(obj, Polygon):
            canvas.add_patch(plt.Polygon(obj._points.transpose([1, 0]), fill=False, **s_styles))

        elif isinstance(obj, Polyline):
            canvas.add_patch(plt.Polygon(obj._points, fill=False, closed=obj.is_closed, **s_styles))

        elif isinstance(obj, Curve):
            canvas.add_patch(plt.Polygon(obj._points, fill=False, closed=obj.is_closed, **s_styles))

        elif isinstance(obj, Rectangle):
            canvas.add_patch(plt.Rectangle((obj._x, obj._y), obj._width, obj._height, fill=False, **s_styles))

        elif isinstance(obj, Circle):
            canvas.add_patch(plt.Circle((obj.x, obj.y), obj.r))

        elif isinstance(obj, Point):
            canvas.scatter(obj.x, obj.y, **s_styles)

        elif isinstance(obj, GeoObject):
            self._draw(canvas, obj.bbox,  styles)

        elif hasattr(obj, "__mesh__"):
            R, Z = obj.__mesh__.points
            value = np.asarray(obj.__value__)
            canvas.contour(R, Z, value,
                           linewidths=s_styles.pop("linewidths", 0.5),
                           levels=s_styles.pop("levels", 10),
                           **s_styles
                           )
        else:
            raise RuntimeError(f"Unsupport type {obj}")

        title_styles = styles.pop("title", False)
        if title_styles:
            if not isinstance(title_styles, dict):
                title_styles = {}

            if isinstance(obj, GeoObject):
                text = obj.name
                pos = obj.bbox.center
            elif hasattr(obj, "__mesh__"):
                text = obj.name
                pos = obj.__mesh__.bbox.center
            else:
                text = str(obj)
                pos = None

            title_styles.setdefault("position", pos)

            self._draw(canvas, text, {f"${self.backend}": title_styles})

            # canvas.text(*pos, text,
            #             horizontalalignment=title_styles.pop('horizontalalignment', 'center'),
            #             verticalalignment=title_styles.pop('verticalalignment', 'center'),
            #             fontsize=title_styles.pop('fontsize', 'xx-small'),
            #             ** title_styles
            #             )

        xlabel = styles.pop("xlabel", None)
        if xlabel is not None:
            canvas.set_xlabel(xlabel)

        ylabel = styles.pop("ylabel", None)
        if ylabel is not None:
            canvas.set_ylabel(ylabel)

    def profiles(self, obj, *args, x_axis=None, x=None, default_num_of_points=128, fontsize=10,  grid=True, signature=None, title=None, **kwargs):

        fontsize = kwargs.pop("fontsize", 10)

        nprofiles = len(obj)

        fig, canves = plt.subplots(ncols=1, nrows=nprofiles, sharex=True,
                                   figsize=(10, 2*nprofiles))

        self.draw(canves, obj,   styles)

        x_label = kwargs.pop("xlabel", "")

        if len(canves) == 1:
            canves[0].set_xlabel(x_label,  fontsize=fontsize)
        else:
            canves[-1].set_xlabel(x_label,  fontsize=fontsize)

        return self._post(fig, **kwargs)

        if not isinstance(profile_list, collections.abc.Sequence):
            profile_list = [profile_list]

        if isinstance(x_axis, collections.abc.Sequence) and not isinstance(x_axis, np.ndarray):
            x_axis, x_label,  *x_opts = x_axis
            x_opts = (x_opts or [{}])[0]
        else:
            x_axis = [0, 1]
            x_label = ""
            x_opts = {}

        if isinstance(x_axis, Function) and x is not None:
            x_axis = x_axis(x)
        elif x is None and isinstance(x_axis, np.ndarray):
            x = x_axis

        if isinstance(x_axis, np.ndarray):
            x_min = x_axis[0]
            x_max = x_axis[-1]
        elif isinstance(x_axis, collections.abc.Sequence) and len(x_axis) == 2:
            x_min, x_max = x_axis
            x_axis = np.linspace(x_min, x_max, default_num_of_points)
        else:
            raise TypeError(x_axis)

        if x is None and isinstance(x_axis, np.ndarray):
            x = x_axis
        elif callable(x_axis) or isinstance(x_axis, Function):
            x_axis = x_axis(x)

        nprofiles = len(profile_list)

        fig, sub_plot = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

        if not isinstance(sub_plot,  (collections.abc.Sequence, np.ndarray)):
            sub_plot = [sub_plot]

        for idx, profile_grp in enumerate(profile_list):

            if not isinstance(profile_grp, list):
                profile_grp = [profile_grp]
            ylabel = None
            for jdx, p_desc in enumerate(profile_grp):
                profile, label, *o_args = p_desc
                opts = {}
                if len(o_args) > 0 and ylabel is None:
                    ylabel = o_args[0]
                if len(o_args) > 1:
                    opts = o_args[1]

                y = None

                if isinstance(profile, Function) or callable(profile):
                    try:
                        y = profile(x)
                    except Exception as error:
                        raise RuntimeError(
                            f"Can not get profile [idx={idx} jdx={jdx}]! name={getattr(profile,'_name',profile)}\n {error} ") from error

                elif isinstance(profile, np.ndarray) and len(profile) == len(x):
                    y = profile
                elif np.isscalar(profile):
                    y = np.full_like(x, profile, dtype=float)
                else:
                    raise RuntimeError(f"Illegal profile! {profile}!={x}")

                if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
                    logger.warning(f"Illegal profile! {(type(x) ,type(y), label, o_args)}")
                    continue
                elif x.shape != y.shape:
                    logger.warning(f"Illegal profile! {x.shape} !={y.shape}")
                    continue
                else:
                    # 删除 y 中的 nan
                    mark = np.isnan(y)
                    # if np.any(mark):
                    #     logger.warning(f"Found NaN in array  {np.argwhere(mark)}! {profile}  ")
                    sub_plot[idx].plot(x_axis[~mark], y[~mark], label=label, **opts)

            sub_plot[idx].legend(fontsize=fontsize)

            if grid:
                sub_plot[idx].grid()

            if ylabel is not None:
                sub_plot[idx].set_ylabel(ylabel, fontsize=fontsize)
            sub_plot[idx].labelsize = "media"
            sub_plot[idx].tick_params(labelsize=fontsize)

        if len(sub_plot) <= 1:
            sub_plot[0].set_xlabel(x_label,  fontsize=fontsize)
        else:
            sub_plot[-1].set_xlabel(x_label,  fontsize=fontsize)

        return fig

    def draw_profile(self, profiles,  x_axis,   canves: plt.Axes = ..., style=None, **kwargs):
        if style is None:
            style = {}

        fontsize = style.get("fontsize", 10)

        ylabel = None

        x_value = x_axis

        if not isinstance(profiles, collections.abc.Sequence):
            profiles = [profiles]

        for profile, label, legend,  *opts in profiles:

            y = None

            if isinstance(profile, Function) or callable(profile):
                try:
                    y = profile(x_value)
                except Exception as error:
                    raise RuntimeError(
                        f"Can not get profile! name={getattr(profile,'name',profile)}\n {error} ") from error

            elif isinstance(profile, array_type) and len(profile) == len(x_value):
                y = profile
            elif np.isscalar(profile):
                y = np.full_like(x_value, profile, dtype=float)
            else:
                raise RuntimeError(f"Illegal profile! {profile}!={x_value}")

            if not isinstance(y, array_type) or not isinstance(x_value, array_type):
                logger.warning(f"Illegal profile! {(type(x_value) ,type(y), label, opts)}")
                continue
            elif x.shape != y.shape:
                logger.warning(f"Illegal profile! {x_value.shape} !={y.shape}")
                continue
            else:
                # 删除 y 中的 nan
                mark = np.isnan(y)
                # if np.any(mark):
                #     logger.warning(f"Found NaN in array  {np.argwhere(mark)}! {profile}  ")
                canves.plot(x_axis[~mark], y[~mark], label=label, **opts)

        canves.legend(fontsize=fontsize)

        if kwargs.get("grid", True):
            canves.grid()

        if ylabel is not None:
            canves.set_ylabel(ylabel, fontsize=fontsize)
        canves.labelsize = "media"
        canves.tick_params(labelsize=fontsize)

    @ staticmethod
    def parse_profile(desc, holder=None, **kwargs):
        opts = {}
        if desc is None:
            return None, {}
        elif isinstance(desc, str):
            data = desc
            opts = {"label": desc}
        elif isinstance(desc, collections.abc.Mapping):
            data = desc.get("name", None)
            if data is None:
                data = desc.get("data", None)
            opts = desc.get("opts", {})
        elif isinstance(desc, tuple):
            data, opts = desc
        elif isinstance(desc, Dict):
            data = desc.data
            opts = desc.opts
        elif isinstance(desc, np.ndarray):
            data = desc
            opts = {}
        else:
            raise TypeError(f"Illegal profile type! {desc}")

        if isinstance(opts, str):
            opts = {"label": opts}

        if opts is None:
            opts = {}

        opts.setdefault("label", "")

        if isinstance(data, str):
            data = try_get(holder, data, None)
        elif isinstance(data, np.ndarray):
            pass
        elif data == None:
            logger.error(f"Value error { (data)}")
        else:
            logger.error(f"Type error {data}")
        return data, opts


__SP_EXPORT__ = MatplotlibView

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
