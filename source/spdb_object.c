#include "spdb_object.h"

int dobj_init(SpDBObject *d, int nd, unsigned int *dimensions, unsigned int *strides, unsigned int element_size, int dtype) { return 0; }
int dobj_free(SpDBObject *d) { return 0; }

int dobj_set_value_by_typename(SpDBObject *d, char const *v, const char *dtype, int nd, unsigned int *dimensions) { return 0; }
int dobj_set_value_by_type(SpDBObject *d, char const *v, const char *dtype, int nd, unsigned int *dimensions) { return 0; }

int dobj_set_int(SpDBObject *d, char const *v, int nd, unsigned int *dimensions) { return 0; }
int dobj_set_char(SpDBObject *d, char const *v, int nd, unsigned int *dimensions) { return 0; }
int dobj_set_double(SpDBObject *d, char const *v, int nd, unsigned int *dimensions) { return 0; }

// switch (dtype)
// {
// case UDA_TYPE_SHORT:
//     success = setReturnDataShortScalar(data_block, static_cast<short>(value.as_int()), NULL);
//     break;
// case UDA_TYPE_LONG:
//     success = setReturnDataLongScalar(data_block, value.as_llong(), NULL);
//     break;
// case UDA_TYPE_FLOAT:
//     success = setReturnDataFloatScalar(data_block, value.as_float(), NULL);
//     break;
// case UDA_TYPE_DOUBLE:
//     success = setReturnDataDoubleScalar(data_block, value.as_double(), NULL);
//     break;
// case UDA_TYPE_INT:
//     success = setReturnDataIntScalar(data_block, value.as_int(), NULL);
//     break;
// case UDA_TYPE_STRING:
//     success = setReturnDataString(data_block, value.as_string(), NULL);
//     break;
// default:
//     success = -1;
//     // throw std::runtime_error("illega data type !");
// }