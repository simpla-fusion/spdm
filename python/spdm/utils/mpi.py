import os

MPI = None
if 'OMPI_COMM_WORLD_SIZE' in os.environ:
    try:
        from mpi4py import MPI 
        # as _MPI   MPI = _MPI
    except ImportError:
        pass
