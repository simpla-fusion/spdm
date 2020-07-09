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

Node::Node(Node *parent, Entry *entry)
    : m_parent_(parent), m_entry_(entry)
{
    m_entry_->bind(this);
}

Node::Node(Node *parent, std::string const &backend)
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
std::any Node::get_scalar() const { return m_entry_->get_scalar(); }

void Node::set_scalar(std::any const &v) { m_entry_->set_scalar(v); }
// void as_scalar(char const *); // set value , if fail then throw exception

std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> Node::get_raw_block() const { return m_entry_->get_raw_block(); }

void Node::set_raw_block(std::shared_ptr<char> const &p, std::type_info const &t, std::vector<size_t> const &d) { m_entry_->set_raw_block(p, t, d); }

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

Node::range Node::children() const
{
    auto r = m_entry_->children();
    return Node::range(Node::iterator(std::get<0>(r), std::get<2>(r)), Node::iterator(std::get<1>(r)));
}

void Node::remove_child(int idx) { return m_entry_->remove_child(idx); }

void Node::remove_child(std::string const &key) { return m_entry_->remove_child(key); }

void Node::remove_children() { return m_entry_->remove_children(); }

//----------------------------------------------------------------------------------------------------------
// function level 1
Node::range Node::select(XPath const &path) const
{
    auto r = m_entry_->select(path);
    return range(Node::iterator(std::get<0>(r), std::get<2>(r)), Node::iterator(std::get<1>(r)));
}

Node &Node::select_one(XPath const &path) { return *m_entry_->select_one(path); }

Node &Node::select_one(XPath const &path) const { return *m_entry_->select_one(path); }

Node &Node::operator[](std::string const &path) { return *m_entry_->select_one(XPath(path)); }

Node &Node::operator[](size_t idx) { return *m_entry_->child(idx); }
