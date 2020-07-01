
#ifndef SPDB_OBJECT_H_
#define SPDB_OBJECT_H_

#ifdef __cplusplus
extern "C"
{
#endif

    struct SpDBObject_
    {
        char *data;
        int nd;
        unsigned int *dimensions;
        unsigned int *strides;
        unsigned int element_size;
        int dtype;
        int flags;
        char tmp[];
    };

    typedef struct SpDBObject_ SpDBObject;

    int dobj_init(SpDBObject *, int nd, unsigned int *dimensions, unsigned int *strides, unsigned int element_size, int dtype);
    int dobj_free(SpDBObject *);

    int dobj_set_value_by_typename(SpDBObject *d, char const *v, const char *dtype, int nd, unsigned int *dimensions);
    int dobj_set_value_by_type(SpDBObject *d, char const *v, const char *dtype, int nd, unsigned int *dimensions);

    int dobj_set_int(SpDBObject *d, char const *v, int nd, unsigned int *dimensions);
    int dobj_set_char(SpDBObject *d, char const *v, int nd, unsigned int *dimensions);
    int dobj_set_double(SpDBObject *d, char const *v, int nd, unsigned int *dimensions);

#ifdef __cplusplus
}
#endif

#endif //SPDB_OBJECT_H_
