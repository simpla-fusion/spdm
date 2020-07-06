#include "SpDBInterface.h"
#include "SpUtil.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace sp;


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


size_t sp::SpDBInterface::attribute_find(const std::string &name) {}
size_t sp::SpDBInterface::attribute_insert(const std::string &name) {}
int sp::SpDBInterface::attribute_remove(size_t) {}
size_t sp::SpDBInterface::attribute_next(size_t) {}

bool sp::SpDBInterface::attribute_equal(size_t, std::any const &) {}
std::any sp::SpDBInterface::attribute_get_value(size_t) {}
size_t sp::SpDBInterface::attribute_set_value(size_t, std::any const &v) {}

SpNode::NodeType sp::SpDBInterface::type() {}

std::any sp::SpDBInterface::get() const {}
void sp::SpDBInterface::set(std::any) const {}

SpDataBlock sp::SpDBInterface::get_block() const {}
void sp::SpDBInterface::set_block(SpDataBlock &&) {}

std::shared_ptr<sp::SpDBInterface> sp::SpDBInterface::next() const {}

//#########################################################################################################
SpXPath::SpXPath(const std::string &path) : m_path_(path) {}
// SpXPath::~SpXPath() = default;
// SpXPath::SpXPath(SpXPath &&) = default;
// SpXPath::SpXPath(SpXPath const &) = default;
// SpXPath &SpXPath::operator=(SpXPath const &) = default;
const std::string &SpXPath::value() const { return m_path_; }

SpXPath SpXPath::operator/(const std::string &suffix) const
{
    return SpXPath(urljoin(m_path_, suffix));
}
SpXPath::operator std::string() const { return m_path_; }
//#########################################################################################################
struct SpNode::pimpl_s
{
    std::shared_ptr<SpDBInterface> m_node_;
};
struct SpAttribute::pimpl_s
{
    SpNode m_node_;
    std::string m_name_;
    size_t m_hid_;
};

SpAttribute::SpAttribute() : m_pimpl_(new pimpl_s) {}
SpAttribute::~SpAttribute() { delete m_pimpl_; }
SpAttribute::SpAttribute(SpAttribute &&other) : SpAttribute(other.m_pimpl_) { other.m_pimpl_ = nullptr; }
SpAttribute::SpAttribute(SpAttribute const &other)
    : m_pimpl_(
          new pimpl_s{
              other.m_pimpl_->m_node_,
              other.m_pimpl_->m_name_,
              other.m_pimpl_->m_hid_}) {}
SpAttribute::SpAttribute(pimpl_s *p) : m_pimpl_(p) {}

std::string SpAttribute::name() const { return m_pimpl_->m_name_; }
std::any SpAttribute::value() const { return std::any(); }
void SpAttribute::swap(this_type &other) { std::swap(m_pimpl_, other.m_pimpl_); }
std::ostream &SpAttribute::repr(std::ostream &os) const { return os; }
bool SpAttribute::same_as(SpAttribute const &other) const
{
    return m_pimpl_->m_node_ == other.m_pimpl_->m_node_ &&
           m_pimpl_->m_hid_ == other.m_pimpl_->m_hid_;
}
SpAttribute SpAttribute::next() const
{
    NOT_IMPLEMENTED;
    return SpAttribute();
}

std::any SpAttribute::get() const
{
    return m_pimpl_->m_node_.m_pimpl_->m_node_->attribute_get_value(m_pimpl_->m_hid_);
}
SpAttribute &SpAttribute::set(std::any const &v)
{
    m_pimpl_->m_node_.m_pimpl_->m_node_->attribute_set_value(m_pimpl_->m_hid_, v);
    return *this;
}

bool SpAttribute::distance(this_type const &other) const
{
    NOT_IMPLEMENTED;
    return false;
}
//--------------------------------------------------------------------------------------------------------------

SpNode::SpNode() : m_pimpl_(new pimpl_s) {}
SpNode::~SpNode() { delete m_pimpl_; }
SpNode::SpNode(SpNode &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }
SpNode::SpNode(SpNode const &other) : m_pimpl_(new pimpl_s{other.m_pimpl_->m_node_}) {}
void SpNode::swap(SpNode &other) { std::swap(m_pimpl_, other.m_pimpl_); }
std::ostream &SpNode::repr(std::ostream &os) const
{
    os << "NOT IMPLEMENTED!" << std::endl;
    return os;
}

SpAttribute SpNode::attribute(const std::string &name) { return SpAttribute(new SpAttribute::pimpl_s{*this, name}); }
SpAttribute SpNode::attribute(const std::string &name) const { return SpAttribute(new SpAttribute::pimpl_s{*this, name}); }
SpRange<SpAttribute> SpNode::attributes() const { return SpRange<SpAttribute>(); }

bool SpNode::same_as(this_type const &other) const { return m_pimpl_->m_node_ == other.m_pimpl_->m_node_; }
bool SpNode::empty() const { return m_pimpl_->m_node_ == nullptr; }

size_t SpNode::size() const
{
    NOT_IMPLEMENTED;
    return true;
}

SpNode::NodeType SpNode::type() const { return m_pimpl_->m_node_ == nullptr ? NodeType::Null : m_pimpl_->m_node_->type(); }

bool SpNode::is_root() const { return parent().empty(); }
bool SpNode::is_leaf() const { return children().size() == 0; }
bool SpNode::distance(this_type const &target) const
{
    NOT_IMPLEMENTED;
    return 0;
}
size_t SpNode::depth() const
{
    NOT_IMPLEMENTED;
    return 0;
}
SpNode SpNode::next() const
{
    SpNode res;
    if (m_pimpl_->m_node_ != nullptr)
    {
        res->m_pimpl_->m_node_ = res->m_pimpl_->m_node_->next();
    }
    return std::move(res);
}
SpNode SpNode::parent() const
{
    NOT_IMPLEMENTED;
    return SpNode();
};
SpNode SpNode::first_child() const
{
    NOT_IMPLEMENTED;
    return SpNode();
};
SpRange<SpNode> SpNode::ancestor() const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpRange<SpNode> SpNode::descendants() const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpRange<SpNode> SpNode::leaves() const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpRange<SpNode> SpNode::children() const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpRange<SpNode> SpNode::slibings() const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpRange<SpNode> SpNode::path(SpNode const target) const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpRange<SpNode> SpNode::select(SpXPath const &path) const
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
};
SpNode SpNode::select_one(SpXPath const &path) const
{
    NOT_IMPLEMENTED;
    return SpNode();
};
// as object
SpNode SpNode::child(std::string const &) const
{
    NOT_IMPLEMENTED;
    return SpNode();
}
SpNode SpNode::child(std::string const &)
{
    NOT_IMPLEMENTED;
    return SpNode();
}
int SpNode::remove_child(std::string const &key)
{
    NOT_IMPLEMENTED;
    return 0;
}

// as array
SpNode SpNode::child(int)
{
    NOT_IMPLEMENTED;
    return SpNode();
}
SpNode SpNode::child(int) const
{
    NOT_IMPLEMENTED;
    return SpNode();
}
SpNode SpNode::insert_before(int pos)
{
    NOT_IMPLEMENTED;
    return SpNode();
}
SpNode SpNode::insert_after(int pos)
{
    NOT_IMPLEMENTED;
    return SpNode();
}
// SpNode SpNode::prepend() { return insert_before(0); }
// SpNode SpNode::append() { return insert_after(-1); }
int SpNode::remove_child(int idx)
{
    NOT_IMPLEMENTED;
    return 0;
}
size_t SpRange<SpNode>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
SpNode SpRange<SpNode>::begin() const
{
    NOT_IMPLEMENTED;
    return SpNode();
}
SpNode SpRange<SpNode>::end() const
{
    NOT_IMPLEMENTED;
    return SpNode();
}

SpRange<SpNode> SpRange<SpNode>::filter(filter_type const &)
{
    NOT_IMPLEMENTED;
    return SpRange<SpNode>();
}

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
