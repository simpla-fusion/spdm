#include <string>
#include <stdlib.h>

#include "spdb_object.h"

class SpDBMapper
{
public:
    SpDBMapper(std::string const &prefix = "");
    ~SpDBMapper();

    int init();
    int reset();
    int close();

    int fetch(std::string const &request, SpDBObject *data_block) const;
    int update(std::string const &request, SpDBObject *data_block) const;
    int modify(std::string const &request) const;

    static SpDBMapper const &load(std::string const &prefix);

private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};