#ifndef SPDB_H_
#define SPDB_H_
#include <stdlib.h>
#include <string>

#include "SpDBDocument.h"



class SpDB
{
public:
    SpDB();
    ~SpDB();

    int connect(std::string const &connection, std::string const &schema = "");
    int disconnect();

    SpDBDocument create(SpDBOID const &oid);
    SpDBDocument open(SpDBOID const &oid);
    int insert(SpDBOID const &oid, SpDBDocument &&);
    int insert(SpDBOID const &oid, SpDBDocument const &);
    int remove(SpDBOID const &oid);
    int remove(std::string const &query);
    std::vector<SpDBDocument> search(std::string const &query);

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

#endif //SPDB_H_