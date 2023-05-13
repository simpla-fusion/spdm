from .Mesh import Mesh
from .StructuredMesh import StructuredMesh


@Mesh.register("uniform")
class UniformMesh(StructuredMesh):
    pass


__SP_EXPORT__ = UniformMesh
