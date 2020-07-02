
#include "SpDB.h"
#include "SpUtil.h"

// SpProxy const &SpProxy::load(std::string const &prefix)
// {
//     static std::map<std::string, std::shared_ptr<SpProxy>> mappers;

//     auto it = mappers.find(prefix);
//     if (it == mappers.end())
//     {
//         return *mappers.emplace(prefix, std::make_shared<SpProxy>(prefix)).first->second;
//     }
//     else
//     {
//         return *it->second;
//     }
// }
struct SpDB::pimpl_s
{
    // SpProxy proxy;
};

SpDB::SpDB() : m_pimpl_(new pimpl_s){
                   // this->m_pimpl_->proxy.init();
               };

SpDB::~SpDB() { delete this->m_pimpl_; }

int SpDB::connect(std::string const &connection, std::string const &schema)
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
        throw std::runtime_error("Can not load config file from " + urljoin(base, connection) + "! " + error.what());
    }

    // std::string schema = config.first_child().name();
    // if (schema == "mapping")
    // {
    // }
    return 0;
}

int SpDB::disconnect() { return 0; }

SpDocument SpDB::create(SpDocument::id_type const &oid)
{
    SpDocument doc;
    return std::move(doc);
}
SpDocument SpDB::open(SpDocument::id_type const &oid)
{
    SpDocument doc;
    return std::move(doc);
}
int SpDB::insert(SpDocument::id_type const &oid, SpDocument &&) { return 0; }
int SpDB::insert(SpDocument::id_type const &oid, SpDocument const &) { return 0; }
int SpDB::remove(SpDocument::id_type const &oid) { return 0; }
int SpDB::remove(std::string const &query) { return 0; }

std::vector<SpDocument> SpDB::search(std::string const &query)
{
    std::vector<SpDocument> nodes;
    return std::move(nodes);
}
