#ifndef SPDB_HANDLER_H_
#define SPDB_HANDLER_H_

class SpDBHandler
{

public:
    SpDBHandler() = default;

    int fetch(std::string const &request, SpDBObject *data_block) const;
    int update(std::string const &request, SpDBObject *data_block) const;
    int modify(std::string const &request) const;
}

#endif // SPDB_HANDLER_H_