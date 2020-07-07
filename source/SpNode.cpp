#include "SpDB.h"
#include "SpEntry.h"
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
class SpContent
{
public:
    std::unique_ptr<SpContent> copy() const { return std::unique_ptr<SpContent>(new SpContent(*this)); };
    SpNode::TypeOfNode type() const { return SpNode::TypeOfNode::Null; }; //
};

class SpContentScalar : public SpContent
{
public:
    SpContentScalar() {}
    SpContentScalar(SpContentScalar const &other) : m_value_(other.m_value_) {}

    ~SpContentScalar() {}
    void swap(SpContentScalar &other)
    {
        std::swap(m_value_, other.m_value_);
    }

    SpNode::TypeOfNode type() const { return SpNode::TypeOfNode::Scalar; }

    std::shared_ptr<SpContent> copy() const { return std::make_shared<SpContentScalar>(*this); }

    std::any &value() { return m_value_; }
    std::any const &value() const { return m_value_; }
    void value(std::any const &v) { m_value_ = v; }

private:
    std::any m_value_;
};

class SpEntryInMemory : public SpEntry
{
public:
    SpEntryInMemory() : SpEntry(), m_content_(new SpContent) {}
    SpEntryInMemory(SpEntryInMemory const &other) : SpEntry(other), m_content_(other.m_content_->copy()) {}
    SpEntryInMemory(SpEntryInMemory &&other) : SpEntry(std::forward<SpEntryInMemory>(other)), m_content_(std::move(other.m_content_)) {}
    SpEntryInMemory(std::shared_ptr<SpNode> const &parent) : SpEntry(parent), m_content_(new SpContent) {}
    ~SpEntryInMemory() {}

    void swap(SpEntryInMemory &other)
    {
        SpEntry::swap(other);
        std::swap(m_attributes_, other.m_attributes_);
        std::swap(m_content_, other.m_content_);
    };

    std::shared_ptr<SpEntry> copy() const { return std::make_shared<SpEntryInMemory>(*this); }

    SpNode::TypeOfNode type() const { return m_content_->type(); }; //

    std::map<std::string, std::any> attributes() const { return m_attributes_; }

    std::any attribute(std::string const &name) const { return m_attributes_.at(name); }

    int attribute(std::string const &name, std::any const &v)
    {
        m_attributes_[name] = v;
        return 1;
    }

    int remove_attribute(std::string const &name)
    {
        auto p = m_attributes_.find(name);
        if (p == m_attributes_.end())
        {
            return 0;
        }
        else
        {
            m_attributes_.erase(p);
            return 1;
        }
    }

    void value(std::any const &v) { return dynamic_cast<SpContentScalar>(m_content_.get())->value(v); }; // set value
    std::any &value() = 0;                                                                               // get value
    std::any const &value() const = 0;                                                                   // get value
    range_type children() const;
    node_type first_child() const;
    node_type child(int);
    node_type child(int) const;
    node_type insert_before(int pos);
    node_type insert_after(int pos);
    node_type prepend();
    node_type append();
    int remove_child(int idx);
    node_type child(std::string const &) const;
    node_type child(std::string const &);
    int remove_child(std::string const &key);
    range_type select(SpXPath const &path) const;
    node_type select_one(SpXPath const &path) const;

private:
    std::map<std::string, std::any> m_attributes_;
    std::unique_ptr<SpContent> m_content_;
};

SpEntry::range_type SpEntryInMemory::children() const { return SpEntry::range_type{}; }

std::shared_ptr<SpEntry> SpEntryInMemory::first_child() const { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::child(int) { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::child(int) const { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::insert_before(int pos) { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::insert_after(int pos) { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::prepend() { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::append() { return nullptr; }

int SpEntryInMemory::remove_child(int idx) { return 0; }

std::shared_ptr<SpEntry> SpEntryInMemory::child(std::string const &) const { return nullptr; }

std::shared_ptr<SpEntry> SpEntryInMemory::child(std::string const &) { return nullptr; }

int SpEntryInMemory::remove_child(std::string const &key) { return 0; }

SpEntry::range_type SpEntryInMemory::select(SpXPath const &path) const { return SpEntry::range_type{}; }

std::shared_ptr<SpEntry> SpEntryInMemory::select_one(SpXPath const &path) const { return nullptr; }

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

SpNode::SpNode() : m_entry_(std::make_shared<SpEntryInMemory>()) {}

SpNode::~SpNode() {}

SpNode::SpNode(SpNode &&other) : m_entry_(other.m_entry_) { other.m_entry_.reset(); }

SpNode::SpNode(SpNode const &other) : m_entry_(other.m_entry_->copy()) {}

SpNode &SpNode::operator=(SpNode const &other) { return SpNode(other).swap(*this); }

SpNode &SpNode::swap(SpNode &other)
{
    std::swap(m_entry_, other.m_entry_);
    return *this;
}

SpNode::SpNode(std::shared_ptr<SpEntry> const &entry) : m_entry_(entry) {}

std::map<std::string, std::any> SpNode::attributes() const
{
    if (m_entry_ == nullptr)
    {
        return std::map<std::string, std::any>{};
    }
    else
    {
        return m_entry_->attributes();
    }
}

std::any SpNode::attribute(std::string const &name) const { return m_entry_->attribute(name); }

int SpNode::attribute(std::string const &name, std::any const &v) { return m_entry_->attribute(name, v); }

int SpNode::remove_attribute(std::string const &name) { return m_entry_->remove_attribute(name); }

std::ostream &SpNode::repr(std::ostream &os) const { return os; }

bool SpNode::same_as(this_type const &other) const { return m_entry_ == other.m_entry_; }

bool SpNode::empty() const { return m_entry_ == nullptr; }

SpNode::TypeOfNode SpNode::type() const { return m_entry_ == nullptr ? TypeOfNode::Null : m_entry_->type(); }

bool SpNode::is_root() const { return m_entry_ == nullptr ? true : m_entry_->parent() == nullptr; }

bool SpNode::is_leaf() const { return !(type() == TypeOfNode::Object || type() == TypeOfNode::List); }

size_t SpNode::depth() const { return m_entry_ == nullptr ? 0 : m_entry_->depth(); }

SpNode SpNode::parent() const { return SpNode(m_entry_ == nullptr ? nullptr : m_entry_->parent()); }

SpNode SpNode::first_child() const { return SpNode(m_entry_->first_child()); }

// SpNode::range SpNode::children() const { return dynamic_cast<SpNodeImplContainer *>(m_pimpl_.get())->children(); }

// SpNode::range SpNode::select(SpXPath const &selector) const
// {
//     return std::move(range(dynamic_cast<SpNodeImplObject *>(m_pimpl_.get())->select(selector)));
// }

SpNode SpNode::select_one(SpXPath const &selector) const
{
    return SpNode(m_entry_->select_one(selector));
}

SpNode SpNode::child(std::string const &key) const
{
    return SpNode(m_entry_->child(key));
}

SpNode SpNode::child(std::string const &key)
{
    return SpNode(m_entry_->child(key));
}

SpNode SpNode::child(int idx)
{
    return SpNode(m_entry_->child(idx));
}

SpNode SpNode::child(int idx) const
{
    return SpNode(m_entry_->child(idx));
}

// SpNode SpNode::insert_before(int idx)
// {
//     return SpNode(m_entry_->insert_before(idx));
// }

// SpNode SpNode::insert_after(int idx)
// {
//     return SpNode(m_entry_->insert_after(idx));
// }

int SpNode::remove_child(int idx)
{
    return m_entry_->remove_child(idx);
}

int SpNode::remove_child(std::string const &key)
{
    return m_entry_->remove_child(key);
}

//----------------------------------------------------------------------------------------------------------
// level 2

ptrdiff_t SpNode::distance(this_type const &target) const { return path(target).size(); }

SpNode::range SpNode::ancestor() const
{
    NOT_IMPLEMENTED;
    return range(nullptr, nullptr);
}

SpNode::range SpNode::descendants() const
{
    NOT_IMPLEMENTED;
    return range(nullptr, nullptr);
}

SpNode::range SpNode::leaves() const
{
    NOT_IMPLEMENTED;
    return range(nullptr, nullptr);
}

SpNode::range SpNode::slibings() const
{
    NOT_IMPLEMENTED;
    return range(nullptr, nullptr);
}

SpNode::range SpNode::path(SpNode const &target) const
{
    NOT_IMPLEMENTED;
    return range(nullptr, nullptr);
}
//----------------------------------------------------------------------------------------------------------
struct SpNode::iterator::pimpl_s
{
};
SpNode::iterator::iterator() : m_pimpl_(new pimpl_s) {}
SpNode::iterator::~iterator() {}
SpNode::iterator::iterator(iterator const &other) : m_pimpl_(new pimpl_s) {}
SpNode::iterator::iterator(iterator &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}
SpNode::iterator &SpNode::iterator::swap(iterator &other) { return *this; }
SpNode::iterator SpNode::iterator::next() const { return iterator(); }
bool SpNode::iterator::equal(iterator const &) const { return false; }
ptrdiff_t SpNode::iterator::distance(iterator const &) const { return 0; }
SpNode::iterator::pointer SpNode::iterator::self() { return nullptr; }