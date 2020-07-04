#include "SpDocument.h"
#include "SpUtil.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
namespace sp
{

    // SpNode::Backend::Backend() {}
    // SpNode::Backend::~Backend() {}
    // void SpNode::Backend::set_attribute(std::string const &name, std::any const &v) {}
    // std::any SpNode::Backend::get_attribute(std::string const &name) { return std::any(nullptr); }
    // void SpNode::Backend::remove_attribute(std::string const &name) {}

    //#########################################################################################################

    SpXPath::SpXPath(std::string const &path) : m_path_(path) {}
    SpXPath::SpXPath(const char *path) : m_path_(path) {}
    // SpXPath::~SpXPath() = default;
    // SpXPath::SpXPath(SpXPath &&) = default;
    // SpXPath::SpXPath(SpXPath const &) = default;
    // SpXPath &SpXPath::operator=(SpXPath const &) = default;
    std::string const &SpXPath::value() const { return m_path_; }

    SpXPath SpXPath::operator/(std::string const &suffix) const
    {
        return SpXPath(urljoin(m_path_, suffix));
    }
    SpXPath::operator std::string() const { return m_path_; }
    //#########################################################################################################
    struct SpNode ::pimpl_s
    {
        void set_attribute(std::string const &name, std::any const &v) {}
        std::any get_attribute(std::string const &name) { return std::any(nullptr); }
        void remove_attribute(std::string const &name) {}
    };
    SpNode::~SpNode() { delete m_pimpl_; }
    SpNode::SpNode(SpNode &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }
    SpNode::SpNode(SpNode *parent) : m_parent_(parent), m_pimpl_(new pimpl_s) {}

    std::ostream &SpNode::repr(std::ostream &os) const
    {
        os << "NOT IMPLEMENTED!" << std::endl;
        return os;
    }

    SpNode::Attribute SpNode::attribute(std::string const &name) const { return SpNode::Attribute(this, name); }

    SpRange<SpNode::Attribute> SpNode::attributes() const
    {
        return SpRange<SpNode::Attribute>();
    }

    SpNode::Iterator SpNode::parent() const
    {
        if (m_parent_ == nullptr)
        {
            throw std::runtime_error("parent node is null!");
        }
        return m_parent_;
    }

    SpNode::Iterator SpNode::first_child() const { return SpNode::Iterator(); }

    SpNode::Range SpNode::children() const
    {
        SpNode::Range range;
        return std::move(range);
    }

    SpNode::Range SpNode::slibings() const
    {
        return this->parent()->children();
    }

    SpNode::Range SpNode::select(SpXPath const &path) const
    {
        SpNode::Range r;
        return std::move(r);
    }

    //----------------------------------------------------------------------------------------------------------

    SpNode::Attribute::Attribute(SpNode const *p, std::string const &name) : m_node_(p), m_name_(name) { ; }
    SpNode::Attribute::~Attribute() {}
    SpNode::Attribute::Attribute(Attribute &&other) : m_node_(other.m_node_), m_name_(other.m_name_)
    {
        other.m_node_ = nullptr;
        other.m_name_ = "";
    }
    SpNode::Attribute::Attribute(Attribute const &other) : m_node_(other.m_node_), m_name_(other.m_name_) {}

    SpNode::Attribute *next(SpNode::Attribute *)
    {
        return nullptr;
    }
    std::string SpNode::Attribute::name() const { return m_name_; }
    std::any SpNode::Attribute::value() const { return get(); }
    bool SpNode::Attribute::same_as(Attribute const &other) const { return false; }
    size_t SpNode::Attribute::distance(Attribute const &other) const { return 0; }

    std::any SpNode::Attribute::get() const
    {
        return (m_node_ == nullptr && m_node_->m_pimpl_ != nullptr) ? nullptr : m_node_->m_pimpl_->get_attribute(m_name_);
    }
    void SpNode::Attribute::set(std::any const &v)
    {

        if (m_node_ == nullptr && m_node_->m_pimpl_ != nullptr)
        {
            m_node_->m_pimpl_->set_attribute(m_name_, v);
        }
    }
    //----------------------------------------------------------------------------------------------------------

    // SpNode::iterator next(SpNode const &n) { return n.next(); }
    // bool same_as(SpNode const &first, SpNode const &second) { return first.same_as(second); }
    // ptrdiff_t distance(SpNode const &first, SpNode const &second) { return first.distance(second); }

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

    SpNode::Iterator SpDocument::root() const { return SpNode::Iterator(m_pimpl_->m_root_); }

    int SpDocument::load(std::string const &path)
    {
        std::ifstream fid(path);
        int res = this->load(fid);
        return res;
    }
    int SpDocument::save(std::string const &path)
    {
        std::ofstream fid(path);
        int res = this->save(fid);
        return res;
    }
    int SpDocument::load(std::istream const &) { return 0; }
    int SpDocument::save(std::ostream const &) { return 0; }
} // namespace sp