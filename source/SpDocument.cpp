#include "SpDocument.h"
#include "SpNdArray.h"
#include "SpUtil.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
namespace sp
{

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
    struct SpNode ::pimpl_s
    {
        SpNode *m_parent_;

        size_t attribute_find(const std::string &name) { return 0; }
        size_t attribute_insert(const std::string &name) { return 0; }
        int attribute_remove(size_t) { return 0; }
        size_t attribute_next(size_t) { return 0; }

        bool attribute_equal(size_t, std::any const &) { return false; }
        std::any attribute_get_value(size_t) { return 0; }
        size_t attribute_set_value(size_t, std::any const &v) { return 0; }

        std::any get() const;
        void set(std::any) const;

        SpNdArray get_block() const;
        void set_block(SpNdArray &&);
    };

    SpNode::SpNode() {}
    SpNode::~SpNode() { delete m_pimpl_; }
    SpNode::SpNode(SpNode &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }
    SpNode::SpNode(SpNode const &other) : m_pimpl_(new pimpl_s{other.m_pimpl_->m_parent_}) {}

    SpNode::SpNode(SpNode *parent) : m_pimpl_(new pimpl_s{parent}) {}

    std::ostream &SpNode::repr(std::ostream &os) const
    {
        os << "NOT IMPLEMENTED!" << std::endl;
        return os;
    }

    SpAttribute SpNode::attribute(const std::string &name) const { return SpAttribute(this, name); }

    SpRange<SpAttribute> SpNode::attributes() const
    {
        return SpRange<SpAttribute>();
    }

    SpNode SpNode::parent() const
    {
        if (m_pimpl_->m_parent_ == nullptr)
        {
            throw std::runtime_error("parent node is null!");
        }
        return *m_pimpl_->m_parent_;
    }

    SpNode SpNode::first_child() const { return SpNode(); }

    SpNode::Range SpNode::children() const
    {
        SpNode::Range range;
        return std::move(range);
    }

    SpNode::Range SpNode::slibings() const
    {
        return this->parent().children();
    }

    SpNode::Range SpNode::select(SpXPath const &path) const
    {
        SpNode::Range r;
        return std::move(r);
    }

    void SpNode::next(){};
    bool SpNode::equal(this_type const &other) const { return true; }
    bool SpNode::distance(this_type const &other) const { return false; }

    //----------------------------------------------------------------------------------------------------------

    struct SpAttribute::pimpl_s
    {
        SpNode const *m_node_;
        std::string m_name_;
        size_t m_hid_;
    };

    SpAttribute::SpAttribute(SpNode const *p, const std::string &name)
        : m_pimpl_(new pimpl_s{p, name, p->m_pimpl_->attribute_find(name)})
    {
    }
    SpAttribute::~SpAttribute() { delete m_pimpl_; }
    SpAttribute::SpAttribute(SpAttribute &&other) : m_pimpl_(other.m_pimpl_) { other.m_pimpl_ = nullptr; }
    SpAttribute::SpAttribute(SpAttribute const &other)
        : m_pimpl_(
              new pimpl_s{
                  other.m_pimpl_->m_node_,
                  other.m_pimpl_->m_name_,
                  other.m_pimpl_->m_hid_}) {}

    std::string SpAttribute::name() const { return m_pimpl_->m_name_; }
    std::any SpAttribute::value() const { return std::any(); }
    void SpAttribute::swap(this_type &other) { std::swap(m_pimpl_, other.m_pimpl_); }
    std::ostream &SpAttribute::repr(std::ostream &os) const { return os; }
    bool SpAttribute::same_as(SpAttribute const &other) const
    {
        return m_pimpl_->m_node_ == other.m_pimpl_->m_node_ &&
               m_pimpl_->m_hid_ == other.m_pimpl_->m_hid_;
    }
    bool SpAttribute::equal(std::any const &value) const
    {
        // p->m_pimpl_->attribute_find(name);
        return m_pimpl_->m_node_ != nullptr &&
               m_pimpl_->m_node_->m_pimpl_->attribute_equal(m_pimpl_->m_hid_, value);
    }
    std::any SpAttribute::get() const
    {
        return (m_pimpl_->m_node_ == nullptr) ? nullptr : m_pimpl_->m_node_->m_pimpl_->attribute_get_value(m_pimpl_->m_hid_);
    }
    SpAttribute &SpAttribute::set(std::any const &v)
    {
        if (m_pimpl_->m_node_ != nullptr)
        {
            m_pimpl_->m_node_->m_pimpl_->attribute_set_value(m_pimpl_->m_hid_, v);
        }
        return *this;
    }

    void SpAttribute::next(){};
    bool SpAttribute::equal(this_type const &other) const { return false; }
    bool SpAttribute::distance(this_type const &other) const { return false; }

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
} // namespace sp