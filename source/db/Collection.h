#ifndef SPDB_COLLECTION_H_
#define SPDB_COLLECTION_H_
#include "Entry.h"
#include <iostream>
#include <memory>
#include <string>

namespace sp::db
{
class Collection
{
public:
    Collection();

    Collection(Collection&&);

    ~Collection();

    Collection(Collection const&) = delete;

    Collection& operator=(Collection const&) = delete;

private:
    struct pimpl_s;
    std::unique_ptr<Entry> m_pimpl_;
};

} // namespace sp::db

#endif // SPDB_COLLECTION_H_