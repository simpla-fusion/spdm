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

    /**
     *  CRUD operations 
     *  erase => delete
     *  
    */
    Entry create(const std::string& request);
    Entry read(const std::string& request);
    void update(const std::string& request, const Entry&);
    void erase(const std::string& request);

private:
    struct pimpl_s;
    std::unique_ptr<Entry> m_pimpl_;
};

} // namespace sp::db

#endif // SPDB_COLLECTION_H_