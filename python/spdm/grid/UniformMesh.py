from .Grid import Grid
from .StructuredMesh import StructuredMesh


@Grid.register("uniform")
class UniformMesh(StructuredMesh):
    pass


__SP_EXPORT__ = UniformMesh
