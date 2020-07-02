#ifndef SPDB_H_
#define SPDB_H_
#include <stdlib.h>
#include <string>

#include "SpDocument.h"



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
    pimpl_s *m_pimpl_;
};

#endif //SPDB_H_