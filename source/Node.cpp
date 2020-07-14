#include "Node.h"
#include "Entry.h"
#include "XPath.h"
#include "utility/Logger.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
namespace sp
{
//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

Node::Node() : m_parent_(nullptr), m_entry_(create_entry()) {}

Node::Node(Node* parent, EntryInterface* entry)
    : m_parent_(parent),
      m_entry_(entry != nullptr ? entry : create_entry()) {}

Node::~Node() {}

Node::Node(this_type const& other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_->copy()) {}

Node::Node(this_type&& other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_.release())
{
    other.m_entry_.reset(other.m_entry_->create());
}

void Node::swap(this_type& other)
{
    std::swap(m_parent_, other.m_parent_);
    std::swap(m_entry_, other.m_entry_);
}

Node& Node::operator=(this_type const& other)
{
    Node(other).swap(*this);
    return *this;
}

Node* Node::copy() const { return new Node(m_parent_, m_entry_->copy()); }

TypeTag Node::type_tag() const { return m_entry_ == nullptr ? TypeTag::Null : m_entry_->type_tag(); }

bool Node::empty() const { return m_entry_ == nullptr; }

bool Node::is_null() const { return type_tag() == TypeTag::Null; }

bool Node::is_scalar() const { return type_tag() == TypeTag::Scalar; }

bool Node::is_block() const { return type_tag() == TypeTag::Block; }

bool Node::is_array() const { return type_tag() == TypeTag::Array; }

bool Node::is_table() const { return type_tag() == TypeTag::Table; }

bool Node::is_root() const { return m_parent_ == nullptr; }

bool Node::is_leaf() const { return !(is_table() || is_array()); }

size_t Node::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

//----------------------------------------------------------------------------------------------------------
// Node: Common
//----------------------------------------------------------------------------------------------------------
Node& Node::parent() const { return *m_parent_; }

void Node::resolve() { m_entry_->resolve(); }

Node* Node::create_child() { return new Node(this, m_entry_->create()); };

Node& Node::as_scalar()
{
    m_entry_->as_scalar();
    return *this;
}

Node& Node::as_block()
{
    m_entry_->as_block();
    return *this;
}

Node& Node::as_array()
{
    m_entry_->as_array();
    return *this;
}

Node& Node::as_table()
{
    m_entry_->as_table();
    return *this;
}

//----------------------------------------------------------------------------------------------------------
// as leaf node,
//----------------------------------------------------------------------------------------------------------
// function level 0

std::any Node::get_scalar() const { return m_entry_->get_scalar(); }

void Node::set_scalar(const std::any& v) { m_entry_->set_scalar(v); }

std::tuple<std::shared_ptr<void>, const std::type_info &, std::vector<size_t>>
Node::get_raw_block() const
{
    return m_entry_->get_raw_block();
}

void Node::set_raw_block(const std::shared_ptr<void>& data /*data pointer*/,
                         const std::type_info& t /*element type*/,
                         const std::vector<size_t>& dims /*dimensions*/)
{
    m_entry_->set_raw_block(data, t, dims);
}

//----------------------------------------------------------------------------------------------------------
// as tree node,
//----------------------------------------------------------------------------------------------------------
// function level 0
size_t Node::size() const { return m_entry_->size(); }

Node::range Node::children() { return m_entry_->children(); }

// Node::const_range Node::children() const { return m_entry_->children(); }

void Node::clear_children() { m_entry_->clear_children(); }

void Node::remove_child(const iterator& it) { m_entry_->remove_child(it); }

void Node::remove_children(const range& r) { m_entry_->remove_children(r); }

Node::iterator Node::begin() { return m_entry_->begin(); }

Node::iterator Node::end() { return m_entry_->end(); }

Node::const_iterator Node::cbegin() const { return m_entry_->cbegin(); }

Node::const_iterator Node::cend() const { return m_entry_->cend(); }

// as table

Node& Node::insert(const std::string& k, const std::shared_ptr<Node>& v)
{
    return *m_entry_->insert(k, v);
}

Node& Node::insert(const std::string& k, const Node& n)
{
    return *m_entry_->insert(k, std::make_shared<Node>(n));
}

Node& Node::insert(const std::string& k, Node&& n)
{
    return *m_entry_->insert(k, std::make_shared<Node>(std::move(n)));
}

Node::range_kv Node::insert(const iterator_kv& b, const iterator_kv& e)
{
    return m_entry_->insert(b, e);
}

Node::range_kv Node::insert(const range_kv& r)
{
    return m_entry_->insert(r.begin(), r.end());
}

// Node::const_range_kv Node::items() const
// {
//     return m_entry_->items();
// }

Node::range_kv Node::items()
{
    return m_entry_->items();
}

Node::iterator Node::find_child(const std::string& k)
{
    return iterator(m_entry_->find_child(k).get());
}

Node& Node::at(const std::string& k)
{
    return *m_entry_->at(k);
}

const Node& Node::at(const std::string& k) const
{
    return *m_entry_->at(k);
}

Node& Node::operator[](const std::string& path) { return at(path); }

const Node& Node::operator[](const std::string& path) const { return at(path); }

// as array

Node& Node::push_back() { return *m_entry_->push_back(); }

Node& Node::push_back(const std::shared_ptr<Node>& n) { return *m_entry_->push_back(n); }

Node& Node::push_back(const Node& n) { return *m_entry_->push_back(n); }

Node& Node::push_back(Node&& n) { return *m_entry_->push_back(std::move(n)); }

Node::range Node::push_back(const range& r) { return m_entry_->push_back(r.begin(), r.end()); }

Node::range Node::push_back(const iterator& b, const iterator& e) { return m_entry_->push_back(b, e); }

Node::iterator Node::find_child(int idx) { return iterator(m_entry_->find_child(idx).get()); }

Node& Node::at(int idx) { return *m_entry_->find_child(idx); }

const Node& Node::at(int idx) const { return *m_entry_->find_child(idx); }

Node& Node::operator[](size_t idx) { return at(idx); }

const Node& Node::operator[](size_t idx) const { return at(idx); }

// as Table
// Node& Node::insert(std::string const&, std::shared_ptr<Node> const&);

// Node& Node::operator[](std::string const& path) { return as_table()[path]; }

// const Node& Node::operator[](std::string const& path) const { return as_table()[path]; }

// // as Array
// Node& Node::push_back(std::shared_ptr<Node> const&);

// Node& Node::operator[](size_t idx) { return as_array()[idx]; }

// const Node& Node::operator[](size_t idx) const { return as_array()[idx]; }
//----------------------------------------------------------------------------------------------------------
// function level 1
Node::range Node::select(XPath const& path)
{
    NOT_IMPLEMENTED;
    return Node::range();
}

Node::const_range Node::select(XPath const& path) const
{
    NOT_IMPLEMENTED;
    return Node::const_range();
}

Node::iterator Node::select_one(XPath const& path)
{
    NOT_IMPLEMENTED;
    return Node::iterator();
}

Node::const_iterator Node::select_one(XPath const& path) const
{
    NOT_IMPLEMENTED;
    return Node::const_iterator();
}

} // namespace sp