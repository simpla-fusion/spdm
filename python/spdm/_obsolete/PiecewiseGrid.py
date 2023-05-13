from __future__ import annotations

from .Mesh import Mesh


class PiecewiseMesh(Mesh):
    def __init__(self, x, *args,    **kwargs) -> None:
        super().__init__(**kwargs)
        self._x = x

    def resample(self, x_min, x_max=None, /, **kwargs):
        x_min = x_min or -np.inf
        x_max = x_max or np.inf
        if x_min <= self.x_min and x_max >= self.x_max:
            return self
        cond_list = []
        func_list = []
        for idx, xp in enumerate(self.x_domain[:-1]):
            if x_max <= xp:
                break
            elif x_min >= self.x_domain[idx+1]:
                continue

            if x_min <= xp:
                cond_list.append(xp)
                func_list.append(self._data[idx])
            if x_max < self.x_domain[idx+1]:
                break
        if len(cond_list) == 0:
            return None
        else:
            cond_list.append(min(x_max, self.x_domain[-1]))
            return PiecewiseFunction(cond_list, func_list)

    def __call__(self, x: typing.Union[float, np.ndarray] = None) -> np.ndarray:
        if x is None:
            x = self._Mesh

        if x is None:
            x = np.linspace(self.x_min, self.x_max, 128)
        elif not isinstance(x, (int, float, np.ndarray)):
            x = np.asarray(x, dtype=float)

        if isinstance(x, np.ndarray) and len(x) == 1:
            x = x[0]

        if isinstance(x, np.ndarray):
            cond_list = [np.logical_and(self.x_domain[idx] <= x, x < self.x_domain[idx+1])
                         for idx in range(len(self.x_domain)-1)]
            cond_list[-1] = np.logical_or(cond_list[-1],
                                          np.isclose(x, self.x_domain[-1]))
            return np.piecewise(x, cond_list, self._data)
        elif isinstance(x, (int, float)):

            if np.isclose(x, self.x_domain[0]):
                idx = 0
            elif np.isclose(x, self.x_domain[-1]):
                idx = -1
            else:
                try:
                    idx = next(i for i, val in enumerate(
                        self.x_domain) if val >= x)-1
                except StopIteration:
                    idx = None
            if idx is None:
                raise ValueError(
                    f"Out of range! {x} not in ({self.x_domain[0]},{self.x_domain[-1]})")

            return self._data[idx](x)
        else:
            raise ValueError(f"Invalid input {x}")


_SP_EXPORT_ = PiecewiseMesh
