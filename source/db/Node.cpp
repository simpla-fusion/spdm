#include "Node.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
#include "Entry.h"
#include "NodePlugin.h"

namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Node& v, int indent = 0, int tab = 4);
std::ostream& fancy_print(std::ostream& os, const sp::db::NodeObject& v, int indent = 0, int tab = 4);
std::ostream& fancy_print(std::ostream& os, const sp::db::NodeArray& v, int indent = 0, int tab = 4);
} // namespace sp::utility

namespace sp::db
{
std::ostream& operator<<(std::ostream& os, Node const& entry) { return sp::utility::fancy_print(os, entry, 0); }
std::ostream& operator<<(std::ostream& os, NodeObject const& node) { return sp::utility::fancy_print(os, node, 0); }
std::ostream& operator<<(std::ostream& os, NodeArray const& node) { return sp::utility::fancy_print(os, node, 0); }
} // namespace sp::db

namespace sp::db
{
//==========================================================================================
// NodeObject
std::shared_ptr<NodeObject> NodeObject::create(const Node& opt)
{
    return create_node_object(opt);
}

//==========================================================================================
// NodeArray

NodeArray::NodeArray(const Node&) : m_container_() {}

NodeArray::NodeArray(const std::initializer_list<Node>& init) : m_container_(init.begin(), init.end()) {}

NodeArray::NodeArray(const NodeArray& other) : m_container_(other.m_container_) {}

NodeArray::NodeArray(NodeArray&& other) : m_container_(std::move(other.m_container_)) {}

void NodeArray::swap(NodeArray& other) { std::swap(m_container_, other.m_container_); }

std::shared_ptr<NodeArray> NodeArray::create(const Node& opt) { return std::make_shared<NodeArray>(opt); }

NodeArray& NodeArray::operator=(const NodeArray& other)
{
    NodeArray(other).swap(*this);
    return *this;
}

size_t NodeArray::size() const { return m_container_.size(); }

void NodeArray::clear() { m_container_.clear(); }

Cursor<Node> NodeArray::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>(); /*(m_container_.begin(), m_container_.end());*/
}

Cursor<const Node> NodeArray::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>(); /*(m_container_.cbegin(), m_container_.cend());*/
}

void NodeArray::for_each(std::function<void(const Node&, Node&)> const& visitor)
{
    for (int i = 0, s = m_container_.size(); i < s; ++i)
    {
        visitor(i, m_container_.at(i));
    }
}

void NodeArray::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
    for (int i = 0, s = m_container_.size(); i < s; ++i)
    {
        visitor(i, m_container_.at(i));
    }
}

Node NodeArray::slice(int start, int stop, int step)
{
    NOT_IMPLEMENTED;
    return Node{};
}

Node NodeArray::slice(int start, int stop, int step) const
{
    NOT_IMPLEMENTED;
    return Node{};
}
bool NodeArray::is_simple() const
{
    auto type = value_type();

    return type != Node::tags::Unknown && type != Node::tags::Object && type != Node::tags::Array && type != Node::tags::Block;
}

int NodeArray::value_type() const
{
    int type = Node::tags::Unknown;
    for (auto const& item : m_container_)
    {
        if (type == Node::tags::Unknown)
        {
            type = item.type();
        }
        else if (type != item.type())
        {
            type = Node::tags::Unknown;
            break;
        }
    }
    return type;
}

void NodeArray::resize(std::size_t num) { m_container_.resize(num); }

Node& NodeArray::insert(int idx, Node const& v)
{
    if (m_container_[idx].value().index() == Node::tags::Null)
    {
        Node(v).swap(m_container_[idx]);
    }
    return m_container_[idx];
}

Node& NodeArray::update(int idx, Node const& v)
{
    Node(v).swap(m_container_[idx]);
    return m_container_[idx];
}

const Node& NodeArray::at(int idx) const { return m_container_.at(idx); }

Node& NodeArray::at(int idx) { return m_container_.at(idx); }

Node& NodeArray::push_back(Node const& node)
{
    m_container_.emplace_back(std::move(node));
    return m_container_.back();
}

Node NodeArray::pop_back()
{
    Node res{m_container_.back().value()};
    m_container_.pop_back();
    return std::move(res);
}

//-----------------------------------------------------------------------------

Node::Node(const std::initializer_list<Node>& init)
{
    bool is_an_object = std::all_of(
        init.begin(), init.end(),
        [](auto&& item) {
            return item.type() == tags::Array && item.as_array().size() == 2 && item.as_array().at(0).type() == tags::String;
        });
    if (is_an_object)
    {
        auto& object_p = this->as_object();

        for (auto& item : init)
        {
            auto& array = item.as_array();

            object_p.update_child(array.at(0).get_value<Node::tags::String>(), array.at(1));
        }
    }
    else if (init.size() == 1)
    {
        m_value_ = init.begin()->value();
    }
    else if (init.size() > 1)
    {
        auto& array_p = this->as_array();
        for (auto&& n : init)
        {
            array_p.push_back(n);
        }
    }
}

Node::Node(char const* c) : m_value_(std::string(c)) {}

Node::Node(Node& other) : m_value_(other.m_value_) {}

Node::Node(const Node& other) : m_value_(other.m_value_) {}

Node::Node(Node&& other) : m_value_(std::move(other.m_value_)) {}

size_t Node::type() const { return m_value_.index(); }

void Node::swap(Node& other) { std::swap(m_value_, other.m_value_); }

void Node::clear() { m_value_.emplace<tags::Null>(); }

NodeArray& Node::as_array()
{
    if (m_value_.index() == tags::Null)
    {
        m_value_.emplace<tags::Array>(NodeArray::create());
    }
    else if (m_value_.index() != tags::Array)
    {
        RUNTIME_ERROR << "illegal type";
    }
    return *std::get<tags::Array>(m_value_);
}

const NodeArray& Node::as_array() const { return *std::get<tags::Array>(m_value_); }

NodeObject& Node::as_object()
{
    if (m_value_.index() == tags::Null)
    {
        m_value_.emplace<tags::Object>(NodeObject::create());
    }
    else if (m_value_.index() != tags::Object)
    {
        RUNTIME_ERROR << "illegal type";
    }
    return *std::get<tags::Object>(m_value_);
}

const NodeObject& Node::as_object() const { return *std::get<tags::Object>(m_value_); }

Path Node::as_path() const
{
    Path path;
    visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& p) { Path(p).swap(path); },
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& spath) { Path(spath).swap(path); },
            [&](const std::variant_alternative_t<Node::tags::Integer, Node::value_type>& idx) { Path{idx}.swap(path); },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        });
    return std::move(path);
}

} // namespace sp::db

namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::NodeObject& object_p, int indent, int tab)
{
    os << "{";

    object_p.for_each(
        [&](const sp::db::Node& key, sp::db::Node const& value) {
            os << std::endl
               << std::setw((indent + 1) * tab) << " "
               << "\"" << key.get_value<sp::db::Node::tags::String>() << "\" : ";
            fancy_print(os, value, indent + 1, tab);
            os << ",";
        });

    os << std::endl
       << std::setw(indent * tab) << " "
       << "}";
    return os;
}

std::ostream& fancy_print(std::ostream& os, const sp::db::NodeArray& array_p, int indent, int tab)
{
    os << "[";

    array_p.for_each([&](const sp::db::Node&, sp::db::Node const& value) {
        os << std::endl
           << std::setw((indent + 1) * tab) << " ";
        fancy_print(os, value, indent + 1, tab);
        os << ",";
    });

    os << std::endl
       << std::setw(indent * tab) << " "
       << "]";
    return os;
}

std::ostream& fancy_print(std::ostream& os, const sp::db::DataBlock& blj, int indent, int tab)
{
    os << "<DATA BLOCK>";
    return os;
}

std::ostream& fancy_print(std::ostream& os, const sp::db::Node& node, int indent, int tab)
{
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Node::tags::Null, sp::db::Node::value_type>& ele) { os << "<NONE>"; },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) { fancy_print(os, *object_p, indent, tab); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Array, sp::db::Node::value_type>& array_p) { fancy_print(os, *array_p, indent, tab); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Block, sp::db::Node::value_type>& blk) { fancy_print(os, blk, indent + 1, tab); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Path, sp::db::Node::value_type>& path) { fancy_print(os, path.str(), indent + 1, tab); },
            [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); } //
        },
        node.value());

    return os;
}

} // namespace sp::utility