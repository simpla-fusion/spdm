import os

SP_DEBUG = os.environ.get("SP_DEBUG", True)

SP_LABEL = os.environ.get("SP_LABEL", __package__[: __package__.find(".")])

SP_MPI = None

if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0")) > 0:
    try:
        from mpi4py import MPI as SP_MPI
    except ImportError:
        SP_MPI = None
try:
    from ..__version__ import __version__
except ImportError:
    SP_VERSION = "develop"
else:
    SP_VERSION = __version__
