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

// class SpMemoryEntry : public SpEntry
// {
// public:
//     SpMemoryEntry();
//     ~SpMemoryEntry();

//     size_t size() const;
//     size_t depth() const;
//     void remove();
//     node_type parent() const;
//     SpNode::TypeOfNode type() const;
//     node_type copy() const;
//     node_type create(SpNode::TypeOfNode) const;
//     std::map<std::string, std::any> attributes() const;
//     std::any attribute(std::string const &name) const;
//     int attribute(std::string const &name, std::any const &v);
//     int remove_attribute(std::string const &name);
//     void value(std::any const &);
//     std::any value() const;
//     range_type children() const;
//     node_type first_child() const;
//     node_type child(int);
//     node_type child(int) const;
//     node_type insert_before(int pos);
//     node_type insert_after(int pos);
//     node_type prepend();
//     node_type append();
//     int remove_child(int idx);
//     node_type child(std::string const &) const;
//     node_type child(std::string const &);
//     int remove_child(std::string const &key);
//     range_type select(SpXPath const &path) const;
//     node_type select_one(SpXPath const &path) const;
// };

template <>
struct SpEntryT<entry_tag_in_memory>::pimpl_s
{
    pimpl_s *copy() const { return new pimpl_s; }
};

template <>
SpEntryT<entry_tag_in_memory>::SpEntryT() : m_pimpl_(new pimpl_s) {}
template <>
SpEntryT<entry_tag_in_memory>::SpEntryT(SpEntryT const &other) : m_pimpl_(other.m_pimpl_->copy()) {}
template <>
SpEntryT<entry_tag_in_memory>::SpEntryT(SpEntryT &&other) : m_pimpl_(std::move(other.m_pimpl_)) {}
template <>
SpEntryT<entry_tag_in_memory>::~SpEntryT() {}
template <>
size_t SpEntryT<entry_tag_in_memory>::size() const { return 0; }
template <>
size_t SpEntryT<entry_tag_in_memory>::depth() const { return 0; }
template <>
void SpEntryT<entry_tag_in_memory>::remove() {}
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::parent() const { return nullptr; }
template <>
SpNode::TypeOfNode SpEntryT<entry_tag_in_memory>::type() const { return SpNode::TypeOfNode::Null; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::copy() const { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::create(SpNode::TypeOfNode) const { return nullptr; }
template <>
std::map<std::string, std::any> SpEntryT<entry_tag_in_memory>::attributes() const { return std::map<std::string, std::any>{}; }
template <>
std::any SpEntryT<entry_tag_in_memory>::attribute(std::string const &name) const { return nullptr; }
template <>
int SpEntryT<entry_tag_in_memory>::attribute(std::string const &name, std::any const &v) { return 0; }
template <>
int SpEntryT<entry_tag_in_memory>::remove_attribute(std::string const &name) { return 0; }
template <>
void SpEntryT<entry_tag_in_memory>::value(std::any const &) {}
template <>
std::any SpEntryT<entry_tag_in_memory>::value() const { return nullptr; }
template <>
SpEntry::range_type SpEntryT<entry_tag_in_memory>::children() const { return SpEntry::range_type{}; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::first_child() const { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::child(int) { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::child(int) const { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::insert_before(int pos) { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::insert_after(int pos) { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::prepend() { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::append() { return nullptr; }
template <>
int SpEntryT<entry_tag_in_memory>::remove_child(int idx) { return 0; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::child(std::string const &) const { return nullptr; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::child(std::string const &) { return nullptr; }
template <>
int SpEntryT<entry_tag_in_memory>::remove_child(std::string const &key) { return 0; }
template <>
SpEntry::range_type SpEntryT<entry_tag_in_memory>::select(SpXPath const &path) const { return SpEntry::range_type{}; }
template <>
std::shared_ptr<SpEntry> SpEntryT<entry_tag_in_memory>::select_one(SpXPath const &path) const { return nullptr; }

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

SpNode::SpNode() : m_entry_(std::make_shared<SpEntryT<entry_tag_in_memory>>()) {}

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