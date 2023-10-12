from spdm.data.sp_property import AttributeTree
from spdm.data.Entry import open_entry


class Tokamak(AttributeTree):
    pass


# tok = Tokamak({
#     "wall": {
#         "ids_properties": {
#             "comment": "just a test"
#         }

#     },
#     "core_transport": {"model": [{"identifier": "first"}, {"identifier": "second"}]}
# })
WORKSPACE = "/home/salmon/workspace"  # "/ssd01/salmon_work/workspace/"

tok = Tokamak(open_entry(f"file+GEQdsk://{WORKSPACE}/gacode/neo/tools/input/profile_data/g141459.03890#equilibrium"))

print(tok.time_slice[0].profiles_1d.psi)
