#ifndef SPDB_PROXY_H_
#define SPDB_PROXY_H_
#include <stdlib.h>
#include <string>

#include "SpDBObject.h"

class SpDBProxy
{
public:
    SpDBProxy();
    ~SpDBProxy();

    int init(std::string const &prefix);
    int reset();
    int close();

    int apply(std::string const &request, SpDBObject *out, SpDBObject *in = nullptr) const;
    int fetch(std::string const &request, SpDBObject *data_block) const;
    int update(std::string const &request, SpDBObject *data_block) const;
    int modify(std::string const &request) const;
    int remove(std::string const &request) const;

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

#endif //SPDB_PROXY_H_