
#ifndef SPDB_OBJECT_INTERFACE_H_
#define SPDB_OBJECT_INTERFACE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#define SpObject_INTERFACE_HEAD ;

typedef struct
{
    SpObject_INTERFACE_HEAD;

} SpObjectInterface;

typedef struct
{
    SpObject_INTERFACE_HEAD;

    char *data;
    unsigned int element_size;
    int dtype;
    int nd;
    unsigned int *dimensions;
    unsigned int *strides;
    int flags;
    char _[];
} SpDataBlockInterface;

#ifdef __cplusplus
}
#endif

#endif //SPDB_OBJECT_INTERFACE_H_
