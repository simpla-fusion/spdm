
#include "SpDB.h"
#include "SpDBProxy.h"

#include "SpUtil.h"

// SpDBProxy const &SpDBProxy::load(std::string const &prefix)
// {
//     static std::map<std::string, std::shared_ptr<SpDBProxy>> mappers;

//     auto it = mappers.find(prefix);
//     if (it == mappers.end())
//     {
//         return *mappers.emplace(prefix, std::make_shared<SpDBProxy>(prefix)).first->second;
//     }
//     else
//     {
//         return *it->second;
//     }
// }
struct SpDB::pimpl_s
{
    SpDBProxy proxy;
};

SpDB::SpDB() : m_pimpl_(new pimpl_s)
{
    this->m_pimpl_->proxy.init();
};

SpDB::~SpDB() { delete this->m_pimpl_; }

int SpDB::connect(std::string const &connection, std::string const &schema = "")
{

    const char *config_path = getenv("SPDB_CONFIG_PATH");

    std::string base = config_path == nullptr ? "" : std::string("local://") + config_path;

    SpDocument config;

    try
    {
        config.load(urljoin(urljoin(base, connection), schema + "/config.xml"));
    }
    catch (std::exception const &error)
    {
        throw std::runtime_error("Can not load config file from " + urljoin(prefix, connection) + "! " + error.what());
    }

    std::string schema = config.first_child().name();
    if (schema == "mapping")
    {
    }
}
