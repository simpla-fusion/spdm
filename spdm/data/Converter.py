'''Convert data structr/format  '''
import collections
import functools

import numpy

from spdm.util.data_entry import DataEntry
from spdm.util.entry import Entry
from spdm.util.logger import logger
from spdm.util.plugin_util import Plugins


class ConvertPlugins(Plugins):
    pass


# _plugins = ConvertPlugins(f"{__package__}.plugins.converter")


@functools.singledispatch
def cannonical_data(d):
    # logger.warning(f"Unkown data type {type(d)}")
    return d


@cannonical_data.register(int)
@cannonical_data.register(str)
@cannonical_data.register(float)
def cannonical_map(d):
    return d


@cannonical_data.register(collections.abc.Mapping)
def cannonical_map(d):
    return {k: cannonical_data(v) for k, v in d.items()}


@cannonical_data.register(collections.abc.Sequence)
def cannonical_list(d):
    return [cannonical_data(v) for v in d]


@cannonical_data.register(numpy.ndarray)
def cannonical_list(d):
    return cannonical_data(d.tolist())


@cannonical_data.register(Entry)
def cannonical_entry(d):
    return cannonical_data(d.serialize())
