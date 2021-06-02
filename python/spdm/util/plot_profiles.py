import collections
import getpass
import datetime
import matplotlib.pyplot as plt
from spdm.numlib import np
from spdm.data.Node import Dict
from spdm.data.Function import Function
from spdm.util.logger import logger
from spdm.util.utilities import try_get


# def signaturebar(fig, text, fontsize=10, pad=5, xpos=20, ypos=7.5,
#                  rect_kw={"facecolor": "grey", "edgecolor": None},
#                  text_kw={"color": "w"}):
#     w, h = fig.get_size_inches()
#     height = ((fontsize+2*pad)/72.)/h
#     rect = plt.Rectangle((0, 0), 1, height, transform=fig.transFigure, clip_on=False, **rect_kw)
#     fig.axes[0].add_patch(rect)
#     fig.text(xpos/72./h, ypos/72./h, text, fontsize=fontsize, **text_kw)
#     fig.subplots_adjust(bottom=fig.subplotpars.bottom+height)


def sp_figure_signature(fig, signature=None):
    if signature is False:
        return fig
    elif not isinstance(signature, str):
        signature = f"Create by SpDM at {datetime.datetime.now().isoformat()}. [user: '{getpass.getuser().capitalize()}']"

    # fig.suptitle(signature)
    fig.text(0.5, 0.04, signature, va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
    # plt.xlabel(signature)
    return fig


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


def plot_profiles(profile_list, *args,   x_axis=None, index_slice=None, fontsize=6,  grid=False, signature=None, **kwargs):
    if not isinstance(profile_list, collections.abc.Sequence):
        profile_list = [profile_list]

    # profile_list += args

    if not isinstance(x_axis, np.ndarray):
        x_axis, x_label,  *x_opts = x_axis
        x_opts = (x_opts or [{}])[0]
    else:
        # x_axis = None
        x_label = ""
        x_opts = {}

    nprofiles = len(profile_list)

    fig, sub_plot = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

    if not isinstance(sub_plot,  (collections.abc.Sequence, np.ndarray)):
        sub_plot = [sub_plot]

    for idx, profile_grp in enumerate(profile_list):
        grp_opts = {}
        if not isinstance(profile_grp, list):
            profile_grp = [profile_grp]
        ylabel = None
        for p_desc in profile_grp:
            profile, label, *o_args = p_desc  # parse_profile(p_desc, **kwargs)
            opts = {}
            if len(o_args) >= 1 and ylabel is None:
                ylabel = o_args[0]
            if len(o_args) >= 2:
                opts = o_args[1]

            y = None
            if isinstance(profile, Function):
                if (x_axis is profile.x) or (isinstance(x_axis, Function) and x_axis is profile.x) or len(x_axis) == len(profile):
                    y = np.asarray(profile)
                else:
                    try:
                        y = np.asarray(profile(x_axis))
                    except ValueError as error:
                        logger.error(f"Can not plot profile {label}! : {error}"),
                        continue
            elif isinstance(profile, np.ndarray):
                y = profile
            elif isinstance(profile, (int, float)):
                y = np.full(x_axis.shape, profile)
            elif callable(profile):
                y = profile(x_axis)

            if not isinstance(y, np.ndarray):
                logger.error(f"Illegal profile '{label}' [{type(profile)}]!")
            elif len(y.shape) == 0:
                y = np.full(x_axis.shape, y)
            elif x_axis.shape != y.shape:
                logger.error(f"length of x,y  must be same! [{label}[{type(profile)}] {x_axis.shape}!={y.shape}]")

            if index_slice is not None:
                x = x_axis[index_slice]
                y = y[index_slice]
            else:
                x = x_axis

            if y is not None:
                sub_plot[idx].plot(x, y, label=label, **opts)

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

    fig.align_ylabels()
    fig.tight_layout()

    fig = sp_figure_signature(fig, signature=signature)
    return fig


def sp_figure(obj, *args, signature=None, **kwargs):
    fig = plt.figure()
    if not hasattr(obj, 'plot'):
        raise NotImplementedError(type(obj))
    else:
        obj.plot(fig.gca(), *args, **kwargs)

    fig = sp_figure_signature(fig, signature=signature)
    # fig.tight_layout()
    fig.gca().axis('scaled')

    return fig
