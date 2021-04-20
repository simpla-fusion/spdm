import collections

import matplotlib.pyplot as plt
import numpy as np
from spdm.data.PhysicalGraph import PhysicalGraph
from spdm.numerical.Function import Function
from spdm.util.logger import logger
from spdm.util.utilities import try_get


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
    elif isinstance(desc, PhysicalGraph):
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


def plot_profiles(profile_list, *args,   x_axis=None, index_slice=None, fontsize=6,  grid=False, **kwargs):
    if not isinstance(profile_list, collections.abc.Sequence):
        profile_list = [profile_list]

    # profile_list += args

    if not isinstance(x_axis, np.ndarray):
        x_axis, x_axis_opts = parse_profile(x_axis, **kwargs)
    else:
        x_axis = None
        x_axis_opts = {}

    nprofiles = len(profile_list)

    fig, sub_plot = plt.subplots(ncols=1, nrows=nprofiles, sharex=True, figsize=(10, 2*nprofiles))

    if not isinstance(sub_plot,  (collections.abc.Sequence, np.ndarray)):
        sub_plot = [sub_plot]

    for idx, profile_grp in enumerate(profile_list):
        grp_opts = {}
        if not isinstance(profile_grp, list):
            profile_grp = [profile_grp]

        for p_desc in profile_grp:
            profile, opts = parse_profile(p_desc, **kwargs)
            y = None
            if isinstance(profile, Function):
                if (x_axis is profile.x) or (isinstance(x_axis, Function) and x_axis is profile.x) or len(x_axis) == len(profile):
                    y = profile.view(np.ndarray)
                else:
                    try:
                        y = profile(x_axis).view(np.ndarray)
                    except RuntimeWarning:
                        logger.debug(p_desc),
                        continue
            elif isinstance(profile, np.ndarray):
                y = profile
            elif callable(profile):
                y = profile(x_axis)

            if y is None:
                logger.error(f"Can not plot profile '{opts.get('label','')}'[{type(profile)}]!")
            elif x_axis.shape != y.shape:
                logger.error(
                    f"length of x,y  must be same! [{opts.get('label','')}[{type(profile)}] {x_axis.shape}!={y.shape}]")

            if index_slice is not None:
                x = x_axis[index_slice]
                y = y[index_slice]
            else:
                x = x_axis
            sub_plot[idx].plot(x, y, **opts)

        sub_plot[idx].legend(fontsize=fontsize)

        if grid:
            sub_plot[idx].grid()

        if "ylabel" in grp_opts:
            sub_plot[idx].set_ylabel(grp_opts["ylabel"], fontsize=fontsize).set_rotation(0)
        sub_plot[idx].labelsize = "media"
        sub_plot[idx].tick_params(labelsize=fontsize)

    if len(sub_plot) <= 1:
        sub_plot[0].set_xlabel(x_axis_opts.get("label", ""),  fontsize=fontsize)
    else:
        sub_plot[-1].set_xlabel(x_axis_opts.get("label", ""),  fontsize=fontsize)

    return fig
