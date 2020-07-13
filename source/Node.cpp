#include "Node.h"
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

Node::Node(Node* parent, Entry* entry) : m_parent_(parent), m_entry_(entry != nullptr ? entry : Entry::create()) {}

Node::~Node() {}

Node::Node(this_type const& other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_->copy()) {}

Node::Node(this_type&& other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_.release())
{
    other.m_entry_.reset(Entry::create());
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

Node* Node::create(TypeTag t, Node* parent, std::string const& backend)
{
    return new Node(parent, Entry::create(t, backend));
}

Node* Node::copy() const { return new Node(m_parent_, m_entry_->copy()); }

Node::TypeTag Node::type_tag() const { return m_entry_ == nullptr ? TypeTag::Null : m_entry_->type_tag(); }

bool Node::empty() const { return m_entry_ == nullptr; }

bool Node::is_null() const { return type_tag() == TypeTag::Null; }

bool Node::is_scalar() const { return type_tag() == TypeTag::Scalar; }

bool Node::is_tensor() const { return type_tag() == TypeTag::Tensor; }

bool Node::is_array() const { return type_tag() == TypeTag::Array; }

bool Node::is_table() const { return type_tag() == TypeTag::Table; }

bool Node::is_root() const { return m_parent_ == nullptr; }

bool Node::is_leaf() const { return !(is_table() || is_array()); }

size_t Node::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

//----------------------------------------------------------------------------------------------------------
// Node: Common
//----------------------------------------------------------------------------------------------------------
Node& Node::parent() const { return *m_parent_; }

void Node::resolve() { NOT_IMPLEMENTED; }

//----------------------------------------------------------------------------------------------------------
// as tree node,  need node.type = List || Object
//----------------------------------------------------------------------------------------------------------
// function level 0

Attributes& Node::attributes() { return m_entry_->attributes(); }

const Attributes& Node::attributes() const { return m_entry_->attributes(); }

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

ScalarEntry& Node::as_scalar() { return *dynamic_cast<ScalarEntry*>(m_entry_.get()); }

const ScalarEntry& Node::as_scalar() const { return *dynamic_cast<const ScalarEntry*>(m_entry_.get()); }

TensorEntry& Node::as_tensor() { return *dynamic_cast<TensorEntry*>(m_entry_.get()); }

const TensorEntry& Node::as_tensor() const { return *dynamic_cast<const TensorEntry*>(m_entry_.get()); }

TreeEntry& Node::as_tree() { return *dynamic_cast<TreeEntry*>(m_entry_.get()); }

const TreeEntry& Node::as_tree() const { return *dynamic_cast<const TreeEntry*>(m_entry_.get()); }

ArrayEntry& Node::as_array()
{

    if (m_entry_->type_tag() != TypeTag::Array)
    {
        auto* p = m_entry_->create(TypeTag::Array);

        dynamic_cast<ArrayEntry*>(p)->push_back(std::move(*this));

        m_entry_.reset(p);
    }
    return *dynamic_cast<ArrayEntry*>(m_entry_.get());
}

const ArrayEntry& Node::as_array() const { return *dynamic_cast<const ArrayEntry*>(m_entry_.get()); }

TableEntry& Node::as_table()
{

    if (m_entry_->type_tag() != TypeTag::Table)
    {
        m_entry_.reset(m_entry_->create(TypeTag::Table));
    }

    return *dynamic_cast<TableEntry*>(m_entry_.get());
}

const TableEntry& Node::as_table() const { return *dynamic_cast<const TableEntry*>(m_entry_.get()); }

//----------------------------------------------------------------------------------------------------------
// as tree node,
//----------------------------------------------------------------------------------------------------------
// function level 0
size_t Node::size() const { return as_tree().size(); }

Node::range Node::children() { return as_tree().children(); }

Node::const_range Node::children() const { return as_tree().children(); }

void Node::clear_children() { as_tree().clear_children(); }

void Node::remove_child(iterator const& it) { as_tree().remove_child(it); }

void Node::remove_children(range const& r) { as_tree().remove_children(r); }

std::shared_ptr<Node> Node::find_child(std::string const& key) { return as_table().find_child(key); }

std::shared_ptr<Node> Node::find_child(int idx) { return as_array().find_child(idx); }

Node::iterator Node::begin() { return as_tree().begin(); }

Node::iterator Node::end() { return as_tree().end(); }

Node::const_iterator Node::cbegin() const { return as_tree().cbegin(); }

Node::const_iterator Node::cend() const { return as_tree().cend(); }

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

//===================================================================================================
// Entry
//----------------------------------------------------------------------------------------------------------

Entry* Entry::create(Node::TypeTag t, std::string const& backend)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

} // namespace sp