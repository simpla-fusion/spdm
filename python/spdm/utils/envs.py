import os

SP_DEBUG = os.environ.get("SP_DEBUG", False)

SP_MPI = None

if os.environ.get('OMPI_COMM_WORLD_SIZE', 0) > 0:
    try:
        from mpi4py import MPI
    except ImportError:
        pass
    else:
        SP_MPI = MPI
