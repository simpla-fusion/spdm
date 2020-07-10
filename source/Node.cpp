#include "Node.h"
#include "Entry.h"
#include "Util.h"
#include "XPath.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
using namespace sp;

//----------------------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------------------

Node::Node(Node *parent, Entry *entry) : m_parent_(parent), m_entry_(entry) { m_entry_->bind(this); }

Node::Node(Node *parent, std::string const &backend) : Node(parent, Entry::create(backend)) {}

Node::Node(Node const &other) : Node(other.m_parent_, other.m_entry_->copy()) {}

Node::Node(Node &&other) : Node(other.m_parent_, other.m_entry_.release()) {}

Node::~Node() {}

Node &Node::operator=(this_type const &other)
{
    this_type(other).swap(*this);
    return *this;
}

std::ostream &_repr_as_yaml(std::ostream &os, Node const &n, int indent)
{
    switch (n.type())
    {
    case NodeTag::List:
    {

        for (auto const &item : n.children())
        {

            os << std::endl
               << std::setw(indent * 2) << std::right << "- ";

            _repr_as_yaml(os, item, indent + 1);
        }
    }
    break;
    case NodeTag::Object:
    {
        bool is_first = true;

        for (auto const &item : n.attributes())
        {
            if (is_first)
            {
                is_first = false;
            }
            else
            {
                os << std::endl
                   << std::setw(indent * 2) << std::right << item.first << ": ";
            }

            os << std::any_cast<std::string>(item.second);
        }
        for (auto const &item : n.children())
        {
            if (is_first)
            {
                is_first = false;
            }
            else
            {
                os << std::endl
                   << std::setw(indent * 2) << " ";
            }
            os << item.get_attribute<std::string>("@name") << ":";
            // os << std::setw(indent * 2) << std::right << item.get_attribute<std::string>("@name") << ": ";
            _repr_as_yaml(os, item, indent + 1);
        }
    }
    break;
    case NodeTag::Null:
    case NodeTag::Scalar:
    default:
    {
        os << std::any_cast<std::string>(n.get_scalar());
    }
    }

    return os;
}

std::ostream &Node::repr(std::ostream &os) const { return _repr_as_yaml(os, *this, 0); }

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

std::any Node::get_attribute(std::string const &key) const { return m_entry_->get_attribute(key); }

void Node::set_attribute(std::string const &key, std::any const &v) { m_entry_->set_attribute(key, v); }

void Node::remove_attribute(std::string const &key) { m_entry_->remove_attribute(key); }

Range<Iterator<const std::pair<const std::string, std::any>>> Node::attributes() const { return std::move(m_entry_->attributes()); }

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

Node &Node::append(std::shared_ptr<Node> const &n) { return *m_entry_->append(n); }

void Node::append(const Iterator<std::shared_ptr<Node>> &b,
                  const Iterator<std::shared_ptr<Node>> &e)
{
    m_entry_->append(b, e);
}

void Node::insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &b,
                  Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &e)
{
    m_entry_->insert(b, e);
}

Node::const_range Node::children() const { return const_range(m_entry_->children()); }

Node::range Node::children() { return range(m_entry_->children()); }

void Node::remove_child(int idx) { return m_entry_->remove_child(idx); }

void Node::remove_child(std::string const &key) { return m_entry_->remove_child(key); }

void Node::remove_children() { return m_entry_->remove_children(); }

//----------------------------------------------------------------------------------------------------------
// function level 1
Node::range Node::select(XPath const &path) { return range(m_entry_->select(path)); }

Node::const_range Node::select(XPath const &path) const { return const_range(m_entry_->select(path)); }

Node &Node::select_one(XPath const &path) { return *m_entry_->select_one(path); }

const Node &Node::select_one(XPath const &path) const { return *m_entry_->select_one(path); }

Node &Node::operator[](std::string const &path) { return *m_entry_->select_one(XPath(path)); }

const Node &Node::operator[](std::string const &path) const { return *m_entry_->select_one(XPath(path)); }

Node &Node::operator[](size_t idx) { return *m_entry_->child(idx); }

const Node &Node::operator[](size_t idx) const { return *m_entry_->child(idx); }
