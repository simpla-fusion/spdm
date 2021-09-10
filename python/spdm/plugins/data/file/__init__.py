__path__ = __import__('pkgutil').extend_path(__path__, __name__)


association = {
    "table":    ".plugins.data.file.PluginTable",
    "bin":      ".plugins.data.file.PluginBinary",
    "h5":       ".plugins.data.file.PluginHDF5",
    "hdf5":     ".plugins.data.file.PluginHDF5",
    "nc":       ".plugins.data.file.PluginNetCDF",
    "netcdf":   ".plugins.data.file.PluginNetCDF",
    "namelist": ".plugins.data.file.PluginNamelist",
    "nml":      ".plugins.data.file.PluginNamelist",
    "xml":      ".plugins.data.file.PluginXML",
    "json":     ".plugins.data.file.PluginJSON",
    "yaml":     ".plugins.data.file.PluginYAML",
    "txt":      ".plugins.data.file.PluginTXT",
    "csv":      ".plugins.data.file.PluginCSV",
    "numpy":    ".plugins.data.file.PluginNumPy",
    "gfile":    ".plugins.data.file.PluginGEQdsk",
    "geqdsk":   ".plugins.data.file.PluginGEQdsk",

    "mds": "spdm.plugins.data.db.PluginMDSplus#MDSplusFile",
    "mdsplus": "spdm.plugins.data.db.PluginMDSplus#MDSplusFile",
    # "db.imas":".spdm.plugins.data.db.IMAS#IMASDocument",
}
