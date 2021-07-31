#ifndef SPDB_SCHEMA_H_
#define SPDB_SCHEMA_H_
#include "Entry.h"
namespace sp::db
{
class Schema
{
public:
    Schema();
    ~Schema();

    bool validate(const URI&, const Entry&) const;

private:
    Entry m_root_;
};
} // namespace sp::db

#endif // SPDB_SCHEMA_H_