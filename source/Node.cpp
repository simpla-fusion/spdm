#include "Node.h"
#include "Entry.h"
#include "Util.h"
#include "XPath.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace sp;

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

Node::Node(std::shared_ptr<Node> const &parent, Entry *entry)
    : m_parent_(parent), m_entry_(entry)
{
    m_entry_->bind(this->shared_from_this());
}

Node::Node(std::shared_ptr<Node> const &parent, std::string const &backend)
    : Node(parent, Entry::create(backend)) {}

Node::Node(Node const &other) : Node(other.m_parent_, other.m_entry_->copy()) {}

Node::Node(Node &&other) : Node(other.m_parent_, other.m_entry_.release()) {}

Node::~Node() {}

Node &Node::operator=(this_type const &other)
{
    this_type(other).swap(*this);
    return *this;
}

std::ostream &Node::repr(std::ostream &os) const { return m_entry_->repr(os); }

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

// bool Node::same_as(this_type const &other) const { return this == &other; }

//----------------------------------------------------------------------------------------------------------
// attribute
//----------------------------------------------------------------------------------------------------------
bool Node::has_attribute(std::string const &key) const { return m_entry_->has_attribute(key); }

bool Node::check_attribute(std::string const &key, std::any const &v) const { return m_entry_->check_attribute(key, v); }

std::any Node::attribute(std::string const &key) const { return m_entry_->attribute(key); }

void Node::attribute(std::string const &key, const char *v) { m_entry_->attribute(key, std::any(std::string(v))); }

void Node::attribute(std::string const &key, std::any const &v) { m_entry_->attribute(key, v); }

void Node::remove_attribute(std::string const &key) {}

Range<Iterator<std::pair<std::string, std::any>>> Node::attributes() const
{
    return Range<Iterator<std::pair<std::string, std::any>>>{};
}

//----------------------------------------------------------------------------------------------------------
// as leaf node,  need node.type = Scalar || Block
//----------------------------------------------------------------------------------------------------------
std::any Node::as_scalar() const { return m_entry_->as_scalar(); }

void Node::as_scalar(std::any const &v) { m_entry_->as_scalar(v); }
// void as_scalar(char const *); // set value , if fail then throw exception

Node::block_type Node::as_block() const { return m_entry_->as_block(); }

void Node::as_block(Node::block_type const &b) { m_entry_->as_block(b); }

//----------------------------------------------------------------------------------------------------------
// as tree node,  need node.type = List || Object
//----------------------------------------------------------------------------------------------------------
// function level 0
Node &Node::parent() const { return *m_parent_; }

Node &Node::child(std::string const &key) { return *m_entry_->child(key); }

const Node &Node::child(std::string const &key) const { return *m_entry_->child(key); }

Node &Node::child(int idx) { return *m_entry_->child(idx); }

const Node &Node::child(int idx) const { return *m_entry_->child(idx); }

Node &Node::append() { return *m_entry_->append(); }

Node::range Node::children() const { return m_entry_->children(); }

void Node::remove_child(int idx) { return m_entry_->remove_child(idx); }

void Node::remove_child(std::string const &key) { return m_entry_->remove_child(key); }

void Node::remove_children() { return m_entry_->remove_children(); }

//----------------------------------------------------------------------------------------------------------
// function level 1
Node::range Node::select(XPath const &path) const { return m_entry_->select(path); }

Node &Node::select_one(XPath const &path) { return *m_entry_->select_one(path); }

Node &Node::select_one(XPath const &path) const { return *m_entry_->select_one(path); }

Node &Node::operator[](std::string const &path) { return *m_entry_->select_one(XPath(path)); }

Node &Node::operator[](size_t idx) { return *m_entry_->child(idx); }
