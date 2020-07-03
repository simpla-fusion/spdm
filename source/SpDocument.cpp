#include "SpDocument.h"
#include "SpDocumentBackend.h"
#include "SpUtil.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
namespace sp
{
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

    SpNode::~SpNode() { delete m_pimpl_; }
    SpNode::SpNode(SpNode &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }
    // SpNode::SpNode(SpNode *parent) : m_parent_(parent) {}

    std::ostream &SpNode::repr(std::ostream &os) const
    {
        os << "NOT IMPLEMENTED!" << std::endl;
        return os;
    }

    std::ostream &operator<<(std::ostream &os, SpNode const &d)
    {
        return d.repr(os);
    };

    SpNode::Attribute SpNode::attribute(std::string const &name) const { return SpNode::Attribute(this, name); }

    Range<SpNode::Attribute> SpNode::attributes() const
    {
        return Range<SpNode::Attribute>();
    }

    SpNode::iterator SpNode::parent() const
    {
        if (m_parent_ == nullptr)
        {
            throw std::runtime_error("parent node is null!");
        }
        return m_parent_;
    }

    SpNode::iterator SpNode::next() const { return SpNode::iterator(); }

    SpNode::iterator SpNode::first_child() const { return SpNode::iterator(); }

    SpNode::range SpNode::children() const
    {
        SpNode::range range;
        return std::move(range);
    }

    SpNode::range SpNode::slibings() const
    {
        return this->parent()->children();
    }

    SpNode::range SpNode::select(SpXPath const &path) const
    {
        SpNode::range r;
        return std::move(r);
    }

    //----------------------------------------------------------------------------------------------------------

    SpNode::Attribute::Attribute(SpNode const *p, std::string const &name) : m_node_(p), m_name_(name) { ; }
    SpNode::Attribute::~Attribute() {}
    SpNode::Attribute::Attribute(Attribute &&other) : m_node_(other.m_node_), m_name_(other.m_name_) {}

    std::string SpNode::Attribute::name() const { return m_name_; }
    std::any SpNode::Attribute::value() const { return get(); }

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

    SpNode::iterator next(SpNode const &n) { return n.next(); }
    bool same_as(SpNode const &first, SpNode const &second) { return first.same_as(second); }
    ptrdiff_t distance(SpNode const &first, SpNode const &second) { return first.distance(second); }

    //##########################################################################################

    SpDocument::SpDocument() {}
    SpDocument::~SpDocument() {}
    SpDocument::SpDocument(SpDocument &&other) : m_root_(other.m_root_){};

    SpDocument::OID::OID() : m_id_(0)
    {
        // TODO:  random  init m_id_
    }
    SpDocument::OID::OID(unsigned long id) : m_id_(id) {}

    SpNode const &SpDocument::root() const { return *m_root_; }

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