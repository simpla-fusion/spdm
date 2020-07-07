#include "Node.h"
#include "Util.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace sp;
//#########################################################################################################

XPath::XPath(const std::string &path) : m_path_(path) {}
// XPath::~XPath() = default;
// XPath::XPath(XPath &&) = default;
// XPath::XPath(XPath const &) = default;
// XPath &XPath::operator=(XPath const &) = default;
const std::string &XPath::str() const { return m_path_; }

XPath XPath::operator/(const std::string &suffix) const { return XPath(urljoin(m_path_, suffix)); }
XPath::operator std::string() const { return m_path_; }

//----------------------------------------------------------------------------------------------------------
// Attributes
//----------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
Attributes *Attributes::create() { return new Attributes; }

Attributes *Attributes::copy() const { return new this_type(*this); }

Attributes &Attributes::operator=(this_type const &other)
{
    Attributes(other).swap(*this);
    return *this;
};

void Attributes::swap(this_type &) {}

std::any Attributes::get(std::string const &name) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

int Attributes::set(std::string const &name, std::any const &v)
{
    NOT_IMPLEMENTED;
    return 0;
}

int Attributes::remove(std::string const &name)
{
    NOT_IMPLEMENTED;
    return 0;
}

//----------------------------------------------------------------------------------------------------------
// Content
//----------------------------------------------------------------------------------------------------------
Content *Content::create(int tag) { return new Content; }

Content *Content::copy() const { return new this_type(*this); }

Content &Content::operator=(this_type const &other)
{
    Content(other).swap(*this);
    return *this;
};

void Content::swap(Content &) {}

Content *Content::as(int tag) { return this; }

int Content::type() const { return NodeTag::Null; }

//----------------------------------------------------------------------------------------------------------
// Entry
//----------------------------------------------------------------------------------------------------------

Entry *Entry::create(int tag) { return new Entry; }

Entry *Entry::copy() const { return new Entry(*this); }

Entry::Entry(int tag) {}
// : m_attributes_(new Attributes::create()),
//   m_content_(new Content::create(tag)){};

Entry::Entry(Entry const &other)
    : m_content_(other.m_content_->copy()),
      m_attributes_(other.m_attributes_->copy()) {}

Entry::Entry(Entry &&other)
    : m_content_(std::move(other.m_content_)),
      m_attributes_(std::move(other.m_attributes_))
{
    other.m_attributes_.reset();
    other.m_content_.reset();
}

Entry::Entry(Attributes *attr, Content *content) : m_attributes_(attr), m_content_(content){};

Entry::~Entry() {}

Entry &Entry::operator=(this_type const &other)
{
    this_type(other).swap(*this);
    return *this;
}

int Entry::type() const { return m_content_->type(); }

void Entry::swap(this_type &other)
{
    std::swap(m_content_, other.m_content_);
    std::swap(m_attributes_, other.m_attributes_);
}

std::ostream &Entry::repr(std::ostream &os) const
{
    return os;
}

// Entry::iterator Entry::first_child() const
// {
//     NOT_IMPLEMENTED;
//     return iterator();
// }

// Entry::range Entry::children() const
// {
//     NOT_IMPLEMENTED;
//     return Entry::range();
// }

// Entry::range Entry::select(XPath const &selector) const
// {
//     NOT_IMPLEMENTED;
//     return Entry::range();
// }

// std::shared_ptr<Entry> Entry::select_one(XPath const &selector) const
//     NOT_IMPLEMENTED;
// {
//     return nullptr;
// }

// std::shared_ptr<Entry> Entry::child(std::string const &key) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Entry> Entry::child(std::string const &key)
// {

//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Entry> Entry::child(int idx)
// {

//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Entry> Entry::child(int idx) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// // Entry Entry::insert_before(int idx)
// // {
// //     return Entry(m_entry_->insert_before(idx));
// // }

// // Entry Entry::insert_after(int idx)
// // {
// //     return Entry(m_entry_->insert_after(idx));
// // }

// int Entry::remove_child(int idx)
// {
//     NOT_IMPLEMENTED;
//     return 0;
// }

// int Entry::remove_child(std::string const &key)
// {
//     NOT_IMPLEMENTED;
//     return 0;
// }

// //----------------------------------------------------------------------------------------------------------
// // level 2

// ptrdiff_t Entry::distance(this_type const &target) const { return path(target).size(); }

// Entry::range Entry::ancestor() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// Entry::range Entry::descendants() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// Entry::range Entry::leaves() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// Entry::range Entry::slibings() const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// Entry::range Entry::path(Entry const &target) const
// {
//     NOT_IMPLEMENTED;
//     return range(nullptr, nullptr);
// }

// //----------------------------------------------------------------------------------------------------------
// // Content
// //----------------------------------------------------------------------------------------------------------

// class SpContent
// {
// public:
//     std::unique_ptr<SpContent> copy() const { return std::unique_ptr<SpContent>(new SpContent(*this)); };
//     int type() const { return int::Null; }; //
// };

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

Node::Node(std::shared_ptr<Node> const &parent, int tag) : m_parent_(parent), m_entry_(Entry::create(tag)){};

Node::Node(Node const &other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_->copy()) {}

Node::Node(Node &&other) : m_parent_(other.m_parent_), m_entry_(std::move(other.m_entry_)) { other.m_entry_.reset(); }

Node::~Node() {}

Node &Node::operator=(this_type const &other)
{
    this_type(other).swap(*this);
    return *this;
}

std::ostream &Node::repr(std::ostream &os) const { return os; }

std::ostream &operator<<(std::ostream &os, Node const &d) { return d.repr(os); }

void Node::swap(this_type &other)
{
    std::swap(m_parent_, other.m_parent_);
    std::swap(m_entry_, other.m_entry_);
}
int Node::type() const { return m_entry_->type(); }

bool Node::is_root() const { return m_parent_ == nullptr; }

bool Node::is_leaf() const { return !(type() == NodeTag::List || type() == NodeTag::Object); }

size_t Node::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

bool Node::same_as(this_type const &other) const { return this == &other; }

Attributes const &Node::attributes() const { return m_entry_->attributes(); }
Attributes &Node::attributes() { return m_entry_->attributes(); }