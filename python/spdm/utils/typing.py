import numpy as np
import numpy.typing as np_tp
import typing
PrimaryNumericType = int | float | bool | complex
ArrayLike = np_tp.ArrayLike
NDArray = np_tp.NDArray[np.float64]
NumericType = PrimaryNumericType | NDArray
