__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from spdm.SpObject import SpObject

SpObject.association.update({
    ".data.file.table":    ".data.file.PluginTable",
    ".data.file.bin":      ".data.file.PluginBinary",
    ".data.file.h5":       ".data.file.PluginHDF5",
    ".data.file.hdf5":     ".data.file.PluginHDF5",
    ".data.file.nc":       ".data.file.PluginNetCDF",
    ".data.file.netcdf":   ".data.file.PluginNetCDF",
    ".data.file.namelist": ".data.file.PluginNamelist",
    ".data.file.nml":      ".data.file.PluginNamelist",
    ".data.file.xml":      ".data.file.PluginXML",
    ".data.file.json":     ".data.file.PluginJSON",
    ".data.file.yaml":     ".data.file.PluginYAML",
    ".data.file.txt":      ".data.file.PluginTXT",
    ".data.file.csv":      ".data.file.PluginCSV",
    ".data.file.numpy":    ".data.file.PluginNumPy",
    ".data.file.gfile":    ".data.file.PluginGEQdsk",
    ".data.file.geqdsk":   ".data.file.PluginGEQdsk",

    ".data.file.mds": ".data.db.PluginMDSplus#MDSplusFile",
    ".data.file.mdsplus": ".data.db.PluginMDSplus#MDSplusFile",
    # ".data.file.mds": ".data.db.MDSplus#MDSplusDocument",
    # ".data.file.mdsplus": ".data.db.MDSplus#MDSplusDocument",
    # ".data.file.gfile": ".data.file.PluginGEQdsk",
    # ".data.file.geqdsk": ".data.file.PluginGEQdsk",
    # "db.imas":".spdm.plugins.data.db.IMAS#IMASDocument",
})
