#ifndef SPDB_DATABASE_H_
#define SPDB_DATABASE_H_
#include "Collection.h"
#include "Document.h"
#include <vector>
namespace sp::db
{
class DataBase
{
public:
    DataBase();
    ~DataBase();

    int connect(std::string const& connection, std::string const& schema = "");
    int disconnect();

    Collection create(const std::string&);
    Collection open(const std::string&);
    void purge(const std::string&);

    Collection search(std::string const& query);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

} // namespace sp::db

#endif // SPDB_DATABASE_H_