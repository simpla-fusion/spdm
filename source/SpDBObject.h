#ifndef SPDB_OBJEC_H_
#define SPDB_OBJEC_H_
#include "SpDBObjectInterface.h"

class SpDBObject : public SpDataBlockInterface
{
public:
    template <typename... Args>
    int set(Args &&... args) { return 0; }
};
#endif //SPDB_OBJEC_H_