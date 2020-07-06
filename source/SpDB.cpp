#include "SpDB.h"
#include "SpUtil.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace sp;


//#########################################################################################################
SpXPath::SpXPath(const std::string &path) : m_path_(path) {}
// SpXPath::~SpXPath() = default;
// SpXPath::SpXPath(SpXPath &&) = default;
// SpXPath::SpXPath(SpXPath const &) = default;
// SpXPath &SpXPath::operator=(SpXPath const &) = default;
const std::string &SpXPath::str() const { return m_path_; }

SpXPath SpXPath::operator/(const std::string &suffix) const { return SpXPath(urljoin(m_path_, suffix)); }
SpXPath::operator std::string() const { return m_path_; }

//#########################################################################################################

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

SpNode::SpNode(std::shared_ptr<SpEntry> const &entry) : m_entry_(entry) {}

SpNode::~SpNode() {}

SpNode::SpNode(SpNode &&other) : m_entry_(other.m_entry_) { other.m_entry_ = nullptr; }

SpNode::SpNode(SpNode const &other) : m_entry_(other.m_entry_) {}

std::map<std::string, std::any> SpNode::attributes() const { return m_entry_ == nullptr ? std::map<std::string, std::any>{} : m_entry_->attributes(); }

std::any SpNode::attribute(std::string const &name) const { return m_entry_ == nullptr ? nullptr : m_entry_->attribute(name); }

int SpNode::attribute(std::string const &name, std::any const &v) { return m_entry_ == nullptr ? 0 : m_entry_->attribute(name, v); }

int SpNode::remove_attribute(std::string const &name) { return m_entry_ == nullptr ? 0 : m_entry_->remove_attribute(name); }

void SpNode::swap(SpNode &other) { std::swap(m_entry_, other.m_entry_); }

std::ostream &SpNode::repr(std::ostream &os) const { return (m_entry_ == nullptr) ? os : m_entry_->repr(os); }

bool SpNode::same_as(this_type const &other) const { return m_entry_ == other.m_entry_ || m_entry_->same_as(other.m_entry_); }

bool SpNode::empty() const { return m_entry_ == nullptr; }

size_t SpNode::size() const { return m_entry_ == nullptr ? 0 : m_entry_->size(); }

SpNode::TypeOfNode SpNode::type() const { return m_entry_ == nullptr ? TypeOfNode::Null : m_entry_->type(); }

bool SpNode::is_root() const { return m_entry_ == nullptr ? false : m_entry_->is_root(); }

bool SpNode::is_leaf() const { return m_entry_ == nullptr ? false : m_entry_->is_leaf(); }

size_t SpNode::depth() const { return m_entry_ == nullptr ? 0 : m_entry_->depth(); }

SpNode SpNode::next() const { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->next())); }

SpNode SpNode::parent() const { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->parent())); }

SpNode SpNode::first_child() const { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->first_child())); }

SpRange<SpNode> SpNode::children() const { return std::move(m_entry_ == nullptr ? range_type() : range_type(m_entry_->children())); }

SpRange<SpNode> SpNode::select(SpXPath const &selector) const { return std::move(m_entry_ == nullptr ? range_type() : range_type(m_entry_->select(selector.str()))); }

SpNode SpNode::select_one(SpXPath const &selector) const { return std::move(m_entry_ == nullptr ? SpNode() : SpNode(m_entry_->select_one(selector.str()))); }

SpNode SpNode::child(std::string const &key) const { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->child(key))); }

SpNode SpNode::child(std::string const &key) { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->child(key))); }

SpNode SpNode::child(int idx) { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->child(idx))); }

SpNode SpNode::child(int idx) const { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->child(idx))); }

SpNode SpNode::insert_before(int idx) { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->insert_before(idx))); }

SpNode SpNode::insert_after(int idx) { return std::move(SpNode(m_entry_ == nullptr ? nullptr : m_entry_->insert_before(idx))); }

int SpNode::remove_child(int idx) { return (m_entry_ == nullptr) ? 0 : m_entry_->remove_child(idx); }

int SpNode::remove_child(std::string const &key) { return (m_entry_ == nullptr) ? 0 : m_entry_->remove_child(key); }

//----------------------------------------------------------------------------------------------------------
// level 2

ptrdiff_t SpNode::distance(this_type const &target) const { return path(target).size(); }

SpRange<SpNode> SpNode::ancestor() const
{
    NOT_IMPLEMENTED;
    return range_type(nullptr, nullptr);
}

SpRange<SpNode> SpNode::descendants() const
{
    NOT_IMPLEMENTED;
    return range_type(nullptr, nullptr);
}

SpRange<SpNode> SpNode::leaves() const
{
    NOT_IMPLEMENTED;
    return range_type(nullptr, nullptr);
}

SpRange<SpNode> SpNode::slibings() const
{
    NOT_IMPLEMENTED;
    return range_type(nullptr, nullptr);
}

SpRange<SpNode> SpNode::path(SpNode const &target) const
{
    NOT_IMPLEMENTED;
    return range_type(nullptr, nullptr);
}
//----------------------------------------------------------------------------------------------------------
//##########################################################################################

SpDocument::OID::OID() : m_id_(0)
{
    // TODO:  random  init m_id_
}
SpDocument::OID::OID(unsigned long id) : m_id_(id) {}

struct SpDocument::pimpl_s
{
    SpNode *m_root_ = nullptr;
};
SpDocument::SpDocument() : m_pimpl_(new pimpl_s) {}
SpDocument::~SpDocument() { delete m_pimpl_; }
SpDocument::SpDocument(SpDocument &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ == nullptr; };

const SpNode &SpDocument::root() const { return *(m_pimpl_->m_root_); }
SpNode &SpDocument::root() { return *(m_pimpl_->m_root_); }

int SpDocument::load(const std::string &path)
{
    std::ifstream fid(path);
    int res = this->load(fid);
    return res;
}
int SpDocument::save(const std::string &path)
{
    std::ofstream fid(path);
    int res = this->save(fid);
    return res;
}
int SpDocument::load(std::istream const &) { return 0; }
int SpDocument::save(std::ostream const &) { return 0; }

//##################################################################################################3

struct SpDB::pimpl_s
{
    // SpProxy proxy;
};

SpDB::SpDB() : m_pimpl_(new pimpl_s){};

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
