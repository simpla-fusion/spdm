
import os

from spdm.core.AoS import AoS
from spdm.core.Expression import Expression, Variable
from spdm.core.Field import Field
from spdm.core.Function import Function, function_like
from spdm.core.HTree import List
from spdm.core.sp_property import sp_property
from spdm.core.TimeSeries import TimeSeriesAoS
from spdm.geometry.Curve import Curve
from spdm.geometry.GeoObject import GeoObject, GeoObjectSet
from spdm.geometry.Point import Point
from spdm.mesh.Mesh import Mesh
from spdm.mesh.mesh_curvilinear import CurvilinearMesh
from spdm.numlib.contours import _find_contours
from spdm.numlib.optimize import minimize_filter
from spdm.utils.constants import *
from spdm.utils.tags import _not_found_
from spdm.utils.tree_utils import merge_tree_recursive
from spdm.utils.typing import (ArrayLike, ArrayType, NumericType, array_type,
                               scalar_type)
from spdm.view.View import draw_profiles

WORKSPACE = "/home/salmon/workspace"

os.environ["SP_DATA_MAPPING_PATH"] = f"{WORKSPACE}/fytok_data/mapping"

input_path = f"{WORKSPACE}/gacode/neo/tools/input/profile_data"

output_path = f"{WORKSPACE}/output"

if __name__ == "__main__":

    _X = Variable(0, "R")

    x = np.linspace(0, scipy.constants.pi*2, 100)

    draw_profiles(
        [
            ([
                (np.sin(_X), {"label": "sin(x)"}),
                (np.cos(_X), {"label": "cos(x)"})
            ], {"y_label": r"$\chi_{e}$", }),
            ([
                (3*_X, {"label": r"$3\times x$"}),
                (4*_X, {"label": r"$4\times x$"})
            ], {"y_label": r"$\chi_{e}$", }),
        ],
        x_value=x,
        x_label="x",
        title="Demo draw profiles",
        styles={"fontsize": 16},
        output=f"{output_path}/demo_draw_profiles.svg"
    )
