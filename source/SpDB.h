#ifndef SPDB_H_
#define SPDB_H_
#include "SpDocument.h"

namespace sp
{ 
    class SpDB
    {
    public:
        SpDB();
        ~SpDB();

        int connect(std::string const &connection, std::string const &schema = "");
        int disconnect();

        SpDocument create(SpDocument::id_type const &oid);
        SpDocument open(SpDocument::id_type const &oid);
        int insert(SpDocument::id_type const &oid, SpDocument &&);
        int insert(SpDocument::id_type const &oid, SpDocument const &);
        int remove(SpDocument::id_type const &oid);
        int remove(std::string const &query);

        std::vector<SpDocument> search(std::string const &query);

    private:
        struct pimpl_s;
        std::unique_ptr<pimpl_s> m_pimpl_;
    };

} // namespace sp

#endif //SPDB_H_