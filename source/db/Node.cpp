#include "Node.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include "../utility/Path.h"
#include "../utility/fancy_print.h"
#include "Cursor.h"
#include "Entry.h"
#include <any>
#include <array>
#include <map>
#include <vector>
namespace sp::db
{
template <typename... Others>
std::string join_path(const std::string& l, const std::string& r) { return (l == "") ? r : l + "/" + r; }

template <typename... Others>
std::string join_path(const std::string& l, const std::string& r, Others&&... others)
{
    return join_path(join_path(l, r), std::forward<Others>(others)...);
}

Node make_node(const Entry::element& element)
{
    Node res;

    if (element.index() > Node::type_tags::Array)
    {
        // std::visit(element, [&](auto&& v) { res.data() = v; });
    }
    else if (element.index() == Node::type_tags::Array)
    {
        res.data().template emplace<Node::type_tags::Array>(nullptr, std::get<Node::type_tags::Array>(element));
    }
    else if (element.index() == Node::type_tags::Object)
    {
        res.data().template emplace<Node::type_tags::Object>(nullptr, std::get<Node::type_tags::Object>(element));
    }

    return std::move(res);
}

//--------------------------------------------------------------------------------------------------
// Node

Node::Node(const std::string& backend) : base_type(nullptr, "")
{
    data().template emplace<type_tags::Object>(this, Entry::create(backend));
};

Node::Node(Node* parent, const std::string& name) : base_type(parent, name){};

//--------------------------------------------------------------------------------------------------
// Object
using NodeObject = std::variant_alternative_t<Node::type_tags::Object, Node::element>;

template <>
NodeObject::HTContainerProxyObject(node_type* self, const std::shared_ptr<container>& container) : m_self_(self), m_container_(container) {}
template <>
NodeObject::HTContainerProxyObject(this_type&& other) : HTContainerProxyObject(other.m_self_, other.m_container_) {}
template <>
NodeObject::HTContainerProxyObject(const this_type& other) : HTContainerProxyObject(other.m_self_, other.m_container_->copy()) {}
// template <>
// NodeObject::~HTContainerProxyObject() {}

template <>
size_t NodeObject::size() const { return m_container_->size(); }

template <>
void NodeObject::clear() { m_container_->clear(); }

template <>
int NodeObject::count(const std::string& key) const { return m_container_->count(key); }

template <>
NodeObject::cursor
NodeObject::insert(const std::string& path)
{
    return make_cursor(m_container_->insert(path)).map<Node>();
}

template <>
NodeObject::cursor
NodeObject::insert(const Path& path) { return make_cursor(m_container_->insert(path)).map<Node>(); }

template <>
void NodeObject::erase(const std::string& path) { m_container_->erase(path); }

template <>
void NodeObject::erase(const Path& path) { erase(path.str()); }

template <>
NodeObject::cursor
NodeObject::find(const std::string& path) { return make_cursor(m_container_->find(path)).map<Node>(); }

template <>
NodeObject::cursor
NodeObject::find(const Path& path) { return make_cursor(m_container_->find(path)).map<Node>(); }

template <>
NodeObject::const_cursor
NodeObject::find(const std::string& path) const { return make_cursor(m_container_->find(path)).map<const Node>(); }

template <>
NodeObject::const_cursor
NodeObject::find(const Path& path) const { return make_cursor(m_container_->find(path)).map<const Node>(); }

//-----------------------------------------------------------------------------------
// Array
using NodeArray = std::variant_alternative_t<Node::type_tags::Array, Node::element>;

template <>
NodeArray::HTContainerProxyArray(node_type* self, const std::shared_ptr<container>& container) : m_self_(self), m_container_(container) {}
template <>
NodeArray::HTContainerProxyArray(this_type&& other) : HTContainerProxyArray(other.m_self_, other.m_container_) {}
template <>
NodeArray::HTContainerProxyArray(const this_type& other) : HTContainerProxyArray(other.m_self_, other.m_container_->copy()) {}
// template <>
// NodeArray::~HTContainerProxyArray() {}

template <>
size_t NodeArray::size() const { return m_container_->size(); }

template <>
void NodeArray::resize(std::size_t num) { m_container_->resize(num); }

template <>
void NodeArray::clear() { m_container_->clear(); }

template <>
NodeArray::cursor
NodeArray::push_back() { return make_cursor(m_container_->push_back()).map<Node>(); }

template <>
void NodeArray::pop_back() { m_container_->pop_back(); }

template <>
typename Node::cursor::reference
NodeArray::at(int idx) { return make_node(m_container_->at(idx)); }

template <>
typename Node::const_cursor::reference
NodeArray::at(int idx) const { return make_node(m_container_->at(idx)); }

// temp
// //-------------------------------------------------------------------
// // level 2
// size_t Node::depth() const { return m_entry_ == nullptr ? 0 : parent().depth() + 1; }

// size_t Node::height() const
// {
//     NOT_IMPLEMENTED;
//     return 0;
// }

// Node::cursor Node::slibings() const { return Node::cursor{}; }

// Node::cursor Node::ancestor() const
// {
//     NOT_IMPLEMENTED;
//     return Node::cursor{};
// }

// Node::cursor Node::descendants() const
// {
//     NOT_IMPLEMENTED;
//     return Node::cursor{};
// }

// Node::cursor Node::leaves() const
// {
//     NOT_IMPLEMENTED;
//     return Node::cursor{};
// }

// Node::cursor Node::shortest_path(Node const& target) const
// {
//     NOT_IMPLEMENTED;
//     return Node::cursor{};
// }

// ptrdiff_t Node::distance(const this_type& target) const
// {
//     NOT_IMPLEMENTED;
//     return 0;
// }

// Node load(const std::string& uri) { NOT_IMPLEMENTED; }

// void save(const Node&, const std::string& uri) { NOT_IMPLEMENTED; }

// Node load(const std::istream&, const std::string& format) { NOT_IMPLEMENTED; }

// void save(const Node&, const std::ostream&, const std::string& format) { NOT_IMPLEMENTED; }

// std::ostream& fancy_print(std::ostream& os, const Node& entry, int indent = 0, int tab = 4)
// {

//     if (entry.type() == Entry::NodeType::Element)
//     {
//         os << to_string(entry.get_element());
//     }
//     else if (entry.type() == Entry::NodeType::Array)
//     {
//         os << "[";
//         for (auto it = entry.first_child(); !it.is_null(); ++it)
//         {
//             os << std::endl
//                << std::setw(indent * tab) << " ";
//             fancy_print(os, it->value(), indent + 1, tab);
//             os << ",";
//         }
//         os << std::endl
//            << std::setw(indent * tab)
//            << "]";
//     }
//     else if (entry.type() == Entry::NodeType::Object)
//     {
//         os << "{";
//         for (auto it = entry.first_child(); !it.is_null(); ++it)
//         {
//             os << std::endl
//                << std::setw(indent * tab) << " "
//                << "\"" << it->name() << "\" : ";
//             fancy_print(os, it->value(), indent + 1, tab);
//             os << ",";
//         }
//         os << std::endl
//            << std::setw(indent * tab)
//            << "}";
//     }
//     return os;
// }

// std::ostream& operator<<(std::ostream& os, Node const& entry) { return fancy_print(os, entry, 0); }

// Node::cursor::cursor() : m_entry_(nullptr) {}

// Node::cursor::cursor(Entry* p) : m_entry_(p == nullptr ? nullptr : p->shared_from_this()) {}

// Node::cursor::cursor(const std::shared_ptr<Entry>& p) : m_entry_(p) {}

// Node::cursor::cursor(const cursor& other) : m_entry_(other.m_entry_->copy()) {}

// Node::cursor::cursor(cursor&& other) : m_entry_(other.m_entry_) { other.m_entry_.reset(); }

// bool Node::cursor::operator==(cursor const& other) const { return m_entry_ == other.m_entry_ || (m_entry_ != nullptr && m_entry_->same_as(other.m_entry_.get())); }

// bool Node::cursor::operator!=(cursor const& other) const { return !(operator==(other)); }

// Node Node::cursor::operator*() { return Node(m_entry_); }

// bool Node::cursor::is_null() const { return m_entry_ == nullptr || m_entry_->type() == Entry::Null; }

// std::unique_ptr<Node> Node::cursor::operator->() { return std::make_unique<Node>(m_entry_); }

// Node::cursor& Node::cursor::operator++()
// {
//     m_entry_ = m_entry_->next();
//     return *this;
// }

// Node::cursor Node::cursor::operator++(int)
// {
//     Node::cursor res(*this);
//     m_entry_ = m_entry_->next();
//     return std::move(res);
// }

} // namespace sp::db