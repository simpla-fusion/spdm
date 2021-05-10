

from typing import Generic
import collections
import copy
import functools
import inspect
import pprint
import typing
from enum import IntFlag
from functools import cached_property
from typing import (Any, Iterator, Mapping, MutableMapping, MutableSequence,
                    Sequence, TypeVar, Union, get_args)
import bisect

import numpy as np
from numpy.lib.arraysetops import isin

from .Node import _TObject


class AoS(Generic[_TObject]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class SoA(Generic[_TObject]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
