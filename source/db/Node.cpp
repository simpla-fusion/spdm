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

//==========================================================================================
// NodeArray

NodeArray::NodeArray(const NodeArray& other) : m_container_(other.m_container_) {}

NodeArray::NodeArray(NodeArray&& other) : m_container_(std::move(other.m_container_)) {}

void NodeArray::swap(NodeArray& other) { std::swap(m_container_, other.m_container_); }

std::shared_ptr<NodeArray> NodeArray::create(const Node& opt) { return std::make_shared<NodeArray>(); }

NodeArray& NodeArray::operator=(const NodeArray& other)
{
    NodeArray(other).swap(*this);
    return *this;
}

size_t NodeArray::size() const { return m_container_->size(); }

void NodeArray::clear() { m_container_->clear(); }

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

void NodeArray::for_each(std::function<void(int, Node&)> const& visitor)
{
    for (int i = 0, s = m_container_->size(); i < s; ++i)
    {
        visitor(i, m_container_->at(i));
    }
}

void NodeArray::for_each(std::function<void(int, const Node&)> const& visitor) const
{
    for (int i = 0, s = m_container_->size(); i < s; ++i)
    {
        visitor(i, m_container_->at(i));
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

void NodeArray::resize(std::size_t num) { m_container_->resize(num); }

Node& NodeArray::insert(int idx, Node v)
{
    if ((*m_container_)[idx].value().index() == Node::tags::Null)
    {
        (*m_container_)[idx].swap(v);
    }
    return (*m_container_)[idx];
}
Node& NodeArray::update(int idx, Node v)
{
    (*m_container_)[idx].swap(v);
    return (*m_container_)[idx];
}
const Node& NodeArray::at(int idx) const { return m_container_->at(idx); }

Node& NodeArray::at(int idx) { return m_container_->at(idx); }

Node& NodeArray::push_back(Node node)
{
    m_container_->emplace_back(std::move(node));
    return m_container_->back();
}

Node NodeArray::pop_back()
{
    Node res{m_container_->back().value()};
    m_container_->pop_back();
    return std::move(res);
}

//-----------------------------------------------------------------------------

Node::Node(std::initializer_list<Node> init)
{
    bool is_an_object = std::all_of(
        init.begin(), init.end(),
        [](auto&& item) {
            return item.value().index() == tags::Array && item.as_array().size() == 2 && item.as_array().at(0).value().index() == tags::String;
        });
    if (is_an_object)
    {
        m_value_.emplace<Node::tags::Object>(NodeObject::create());
        std::get<Node::tags::Object>(m_value_)->init(init);
    }
    else if (init.size() == 1)
    {
        m_value_ = init.begin()->value();
    }
    else if (init.size() > 1)
    {
        m_value_.emplace<Node::tags::Array>(std::make_shared<NodeArray>(init.begin(), init.end()));
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
} // namespace sp::db

namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::NodeObject& object_p, int indent, int tab)
{
    os << "{";

    object_p.for_each(
        [&](const sp::db::Path::Segment& key, sp::db::Node const& value) {
            os << std::endl
               << std::setw(indent * tab) << " "
               << "\"" << std::get<std::string>(key) << "\" : ";
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

    array_p.for_each([&](const sp::db::Path::Segment&, sp::db::Node const& value) {
        os << std::endl
           << std::setw(indent * tab) << " ";
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
            [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) { fancy_print(os, *object_p, indent + 1, tab); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Array, sp::db::Node::value_type>& array_p) { fancy_print(os, *array_p, indent + 1, tab); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Block, sp::db::Node::value_type>& blk) { fancy_print(os, blk, indent + 1, tab); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Path, sp::db::Node::value_type>& path) { fancy_print(os, path.str(), indent + 1, tab); },
            [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); } //
        },
        node.value());

    return os;
}

} // namespace sp::utility