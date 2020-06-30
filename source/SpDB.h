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

    SpDocument create(SpOID const &oid);
    SpDocument open(SpOID const &oid);
    int insert(SpOID const &oid, SpDocument &&);
    int insert(SpOID const &oid, SpDocument const &);
    int remove(SpOID const &oid);
    int remove(std::string const &query);
    
    std::vector<SpDocument> search(std::string const &query);

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

#endif //SPDB_H_