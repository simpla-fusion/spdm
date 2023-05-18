

import typing

import numpy as np
from scipy import interpolate
from spdm.data.Entry import Entry, as_entry
from spdm.data.File import File
from spdm.utils.logger import logger
import pathlib

def sp_read_geqdsk(file):
    """
    :param file: input file / file path
    :return: profile object
    """

    description = file.read(48)
    idum = int(file.read(4))
    nw = int(file.read(4))
    nh = int(file.read(4))
    file.readline()

    rdim = float(file.read(16))
    zdim = float(file.read(16))
    rcentr = float(file.read(16))
    rleft = float(file.read(16))
    zmid = float(file.read(16))
    file.readline()
    rmaxis = float(file.read(16))
    zmaxis = float(file.read(16))
    simag = float(file.read(16))
    sibry = float(file.read(16))
    bcentr = float(file.read(16))
    file.readline()
    current = float(file.read(16))
    simag = float(file.read(16))
    xdum = float(file.read(16))
    rmaxis = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    zmaxis = float(file.read(16))
    xdum = float(file.read(16))
    sibry = float(file.read(16))
    xdum = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    def _read_data(count, width=16):
        data = []
        for n in range(count):
            data.append(float(file.read(width)))
            if n >= count - 1 or ((n + 1) % 5 == 0):
                file.readline()
        data = np.asarray(data)
        return data

    #
    fpol = _read_data(nw)
    pres = _read_data(nw)
    ffprim = _read_data(nw)
    pprim = _read_data(nw)

    psirz = _read_data(nw * nh).reshape([nh, nw])

    qpsi = _read_data(nw)

    try:
        nbbs = int(file.read(5))
        limitr = int(file.read(5))
        file.readline()

        bbsrz = _read_data(nbbs * 2).reshape([nbbs, 2])
        limrz = _read_data(limitr * 2).reshape([limitr, 2])
    except:
        nbbs = 0
        limitr = 0
        bbsrz = None
        limrz = None

    data = {
        "description": description,
        # "idum": idum,
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "simag": simag,
        "sibry": sibry,
        "bcentr": bcentr,
        "current": current,
        # "simag": simag,
        # "rmaxis": rmaxis,
        # "zmaxis": zmaxis,
        # "sibry": sibry,
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprim": pprim,
        "psirz": psirz,
        "qpsi": qpsi,
        "bbsrz": bbsrz,
        "limrz": limrz,

    }
    return data


def sp_write_geqdsk(p, file):
    """
    :param profile: object

    :param file: file path / file
    :return:
    """

    nw = p["nw"]
    nh = p["nh"]

    file.write("%48s%4i%4i%4i\n" % (p.get("description", "NO DESCRIPTION"), 3, p["nw"], p["nh"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["rdim"], p["zdim"], p["rcentr"], p["rleft"], p["zmid"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["rmaxis"], p["zmaxis"], p["simag"], p["sibry"], p["bcentr"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["current"], p["simag"], 0, p["rmaxis"], 0))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["zmaxis"], 0, p["sibry"], 0, 0))

    def _write_data(d):
        count = len(d)
        for n in range(count):
            file.write("%16.9e" % d[n])
            if (n == count - 1) or ((n + 1) % 5 == 0):
                file.write('\n')
            else:
                file.write(' ')

    _write_data(p["fpol"])
    _write_data(p["pres"])
    _write_data(p["ffprim"])
    _write_data(p["pprim"])
    _write_data(p["psirz"].reshape([nw * nh]))
    _write_data(p["qpsi"])
    file.write("%5i%5i\n" % (p["bbsrz"].shape[0], p["limrz"].shape[0]))
    _write_data(p["bbsrz"].reshape([p["bbsrz"].size]))
    _write_data(p["limrz"].reshape([p["limrz"].size]))

    return


def sp_to_geqdsk(eq: typing.Any,  geqdsk: typing.Optional[Entry] = None, nw=128, nh=128) -> Entry:
    eq = as_entry(eq)

    if geqdsk is None:
        geqdsk = Entry({})

    limiter_r = eq["wall/description_2d/0/limiter/unit/0/outline/r"].__value__()
    limiter_z = eq["wall/description_2d/0/limiter/unit/0/outline/r"].__value__()
    geqdsk["limrz"] = np.append(limiter_r.reshape([1, limiter_r.size]),
                                limiter_z.reshape([1, limiter_z.size]), axis=0).transpose()
    geqdsk["rleft"] = limiter_r.min()
    geqdsk["rmid"] = rmid = 0.5*(limiter_r.max() + limiter_r.min())
    geqdsk["rdim"] = rdim = (limiter_r.max() - limiter_r.min())
    geqdsk["zdim"] = zdim = (limiter_z.max() - limiter_z.min())

    # rdim = 0.0
    # zdim = 0.0
    geqdsk["rcentr"] = eq["equilibrium/boundary/geometric_axis/r"].__value__()
    # rleft = 0.0
    geqdsk["zmid"] = zmid = eq["equilibrium/boundary/geometric_axis/z"].__value__()
    geqdsk["rmaxis"] = eq["equilibrium/global_quantities/magnetic_axis/r"].__value__()
    geqdsk["zmaxis"] = eq["equilibrium/global_quantities/magnetic_axis/z"].__value__()
    geqdsk["simag"] = eq["equilibrium/global_quantities/psi_axis"].__value__()
    geqdsk["sibry"] = eq["equilibrium/global_quantities/psi_boundary"].__value__()
    geqdsk["bcentr"] = eq["equilibrium/vacuum_toroidal_field/b0"].__value__()
    geqdsk["current"] = eq["equilibrium/global_quantities/ip"].__value__()

    # boundary

    rbbs = eq["equilibrium/boundary/outline/r"].__value__()
    zbbs = eq["equilibrium/boundary/outline/z"].__value__()

    geqdsk["bbsrz"] = np.append(rbbs.reshape([1, rbbs.size]), zbbs.reshape([1, rbbs.size]), axis=0).transpose()
    # psi

    grid_r, grid_z = np.mgrid[rmid-rdim/2:rmid + rdim/2: nw * 1j, zmid - zdim / 2: zmid + zdim / 2: nh * 1j]
    # coord_r = np.append(coord_r[:, :], coord_r[:, 0].reshape(coord_r.shape[0], 1), axis=1)
    # coord_z = np.append(coord_z[:, :], coord_z[:, 0].reshape(coord_z.shape[0], 1), axis=1)
    # points = np.append(coord_r.reshape([coord_r.size, 1]), coord_z.reshape([coord_z.size, 1]), axis=1)
    geqdsk["psirz"] = eq["equilibrium/profiles_2d/psi"].query(grid_r, grid_z)
    # psi = np.append(psi[:, :], psi[:, 0].reshape(psi.shape[0], 1), axis=1)
    # values = psi[:coord_r.shape[0], :coord_r.shape[1]].reshape(points.shape[0])
    # psirz = interpolate.griddata(points, values, (grid_r, grid_z), method='cubic').transpose()

    # profile
    psi_norm = np.linspace(0.0, 1.0, nw)
    geqdsk["fpol"] = eq["equilibrium/profiles_1d/f"].query(psi_norm)
    geqdsk["pres"] = eq["equilibrium/profiles_1d/pressure"].query(psi_norm)
    geqdsk["ffprim"] = eq["equilibrium/profiles_1d/f_df_dpsi"].query(psi_norm)
    geqdsk["pprim"] = eq["equilibrium/profiles_1d/dpressure_dpsi"].query(psi_norm)
    geqdsk["qpsi"] = eq["equilibrium/profiles_1d/q"].query(psi_norm)

    return geqdsk
    # return Entry({
    #     "nw": nw,
    #     "nh": nh,
    #     "rdim": rdim,
    #     "zdim": zdim,
    #     "rcentr": rcentr,
    #     "rleft": rleft,
    #     "zmid": zmid,
    #     "rmaxis": rmaxis,
    #     "zmaxis": zmaxis,
    #     "simag": simag,
    #     "sibry": sibry,
    #     "bcentr": bcentr,
    #     "current": current,
    #     "bbsrz": bbsrz,
    #     "psirz": psirz,
    #     "fpol": fpol,
    #     "pres": pres,
    #     "ffprim": ffprim,
    #     "pprim": pprim,
    #     "qpsi": qpsi,
    #     "limrz": limrz

    # })


def sp_from_geqdsk(geqdsk: typing.Any, eq: typing.Optional[Entry] = None) -> Entry:
    """Converts a GEQDSK file to an IMAS equilibrium entry.
    """
    geqdsk = as_entry(geqdsk)

    if eq is None:
        eq = Entry({"time_slice": [{}]})

    # eq.time = 0.0
    eq["vacuum_toroidal_field/r0"] = geqdsk["rcentr"].__value__()
    eq["vacuum_toroidal_field/b0"] = [geqdsk["bcentr"].__value__()]

    # rleft = 0.0

    # eq["global_quantities.magnetic_axis.b_field_tor"] = geqdsk["bcentr"]
    nw = geqdsk["nw"].__value__()
    nh = geqdsk["nh"].__value__()
    rmin = geqdsk["rleft"].__value__()
    rmax = geqdsk["rleft"].__value__() + geqdsk["rdim"].__value__()
    zmin = geqdsk["zmid"].__value__() - geqdsk["zdim"].__value__()/2
    zmax = geqdsk["zmid"].__value__() + geqdsk["zdim"].__value__()/2

    psirz = geqdsk["psirz"].__value__()

    if psirz.shape == (nh, nw):
        psirz = psirz.T
        # logger.warning(f"Transposing psirz from {(nh, nw)} to {(nw,nh)}")

    if psirz.shape != (nw, nh):
        raise ValueError(f"Invalid shape for psirz: {psirz.shape}!={(nw, nh)}")

    eq["time_slice"][-1] = {
        "global_quantities": {"magnetic_axis": {"r": geqdsk["rmaxis"].__value__(),
                                                "z": geqdsk["zmaxis"].__value__()},
                              "psi_axis": geqdsk["simag"].__value__(),
                              "psi_boundary": geqdsk["sibry"].__value__(),
                              "ip": geqdsk["current"].__value__()
                              },
        # boundary
        "boundary": {"outline": {"r": geqdsk["bbsrz"][:, 0].__value__(),
                                 "z": geqdsk["bbsrz"][:, 1].__value__()}},

        # profile 1d
        "profiles_1d": {
            "f": geqdsk["fpol"].__value__(),
            "f_df_dpsi": geqdsk["ffprim"].__value__(),
            "pressure": geqdsk["pres"].__value__(),
            "dpressure_dpsi": geqdsk["pprim"].__value__(),
            "q": geqdsk["qpsi"].__value__(),
            "psi": np.linspace(geqdsk["simag"].__value__(), geqdsk["sibry"].__value__(), nw),
        },
        "profiles_2d": [
            {
                "grid_type": {"name": "rectangular", "index": 1},
                "grid": {"dim1": np.linspace(rmin, rmax, nw),
                         "dim2": np.linspace(zmin, zmax, nh)},
                "psi": psirz
            }]
    }

    return eq


@File.register(["gfile", "GEQdsk"])
class GEQdskFile(File):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self._fid = open(pathlib.Path(self.uri.path).expanduser().resolve(),  mode=self.mode_str)
        except OSError as error:
            raise FileExistsError(f"Can not open file {self.uri}! {error}")
        else:
            logger.debug(f"Open File {self.uri} mode={self.mode}")

    # def flush(self, *args, **kwargs):
    #     if self.mode & File.Mode.write:
    #         self.save(self.path)

    def read(self, lazy=False) -> Entry:
        return sp_from_geqdsk(sp_read_geqdsk(self._fid))

    def write(self, d, *args, **kwargs):
        geqdsk = sp_to_geqdsk(d, *args, **kwargs)
        sp_write_geqdsk(geqdsk, self._fid)


__SP_EXPORT__ = GEQdskFile
