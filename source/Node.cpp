

#include "Node.h"
#include "utility/Logger.h"
#include <any>
#include <map>
#include <memory>
#include <variant>
#include <vector>

namespace sp
{
typedef std::vector<std::shared_ptr<Node>> array_t;
typedef std::map<std::string, std::shared_ptr<Node>> map_t;
struct node_t : public std::variant<scalar_t, array_t, map_t>
{
};
typedef node_t entry_in_memory;

template <>
void NodePolicyBody<entry_in_memory>::resolve() { NOT_IMPLEMENTED; }

//----------------------------------------------------------------
// attributes

template <>
bool NodePolicyAttributes<entry_in_memory>::has_attribute(std::string const& key) const
{
    NOT_IMPLEMENTED;
    return false;
}
template <>
bool NodePolicyAttributes<entry_in_memory>::check_attribute(std::string const& key, scalar_t const& v) const
{
    NOT_IMPLEMENTED;
    return false;
}
template <>
void NodePolicyAttributes<entry_in_memory>::set_attribute(const std::string&, const scalar_t&)
{
    NOT_IMPLEMENTED;
}
template <>
scalar_t NodePolicyAttributes<entry_in_memory>::get_attribute(const std::string&) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
scalar_t NodePolicyAttributes<entry_in_memory>::get_attribute(std::string const& key, scalar_t const& default_value)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
void NodePolicyAttributes<entry_in_memory>::remove_attribute(const std::string&)
{
    NOT_IMPLEMENTED;
}
template <>
Range<Iterator<std::pair<std::string, scalar_t>>>
NodePolicyAttributes<entry_in_memory>::attributes() const
{
    NOT_IMPLEMENTED;
    return Range<Iterator<std::pair<std::string, scalar_t>>>{};
}

template <>
void NodePolicyAttributes<entry_in_memory>::clear_attributes()
{
    NOT_IMPLEMENTED;
}

//-------------------------------------------------------------------------------------------------------
// as scalar
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Scalar>::as_interface(TypeTag tag) { return this->self(); }

template <>
void NodePolicy<entry_in_memory, TypeTag::Scalar>::set_scalar(scalar_t const& v) { entry()->emplace<scalar_t>(v); }

template <>
scalar_t NodePolicy<entry_in_memory, TypeTag::Scalar>::get_scalar() const { return std::get<scalar_t>(*entry()); }

//-------------------------------------------------------------------------------------------------------
// as block
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Block>::as_interface(TypeTag tag)
{
    if (tag == TypeTag::Block)
    {
        return this->self();
    }
    else
    {
        NOT_IMPLEMENTED;
        return nullptr;
    }
}
template <>
std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>
NodePolicy<entry_in_memory, TypeTag::Block>::get_raw_block() const
{
    NOT_IMPLEMENTED;
    return std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>{nullptr, typeid(nullptr_t), {}};
}
template <>
void NodePolicy<entry_in_memory, TypeTag::Block>::set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                                                                const std::type_info& /*element type*/,
                                                                const std::vector<size_t>& /*dimensions*/)
{
    NOT_IMPLEMENTED;
}
//-------------------------------------------------------------------------------------------------------
// array
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Array>::as_interface(TypeTag tag) { return this->self(); }
template <>
size_t NodePolicy<entry_in_memory, TypeTag::Array>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Node::range NodePolicy<entry_in_memory, TypeTag::Array>::children()
{
    NOT_IMPLEMENTED;
    return Node::range{};
}
template <>
Node::const_range NodePolicy<entry_in_memory, TypeTag::Array>::children() const
{
    NOT_IMPLEMENTED;
    return Node::const_range{};
}
template <>
void NodePolicy<entry_in_memory, TypeTag::Array>::clear_children() { NOT_IMPLEMENTED; }
template <>
void NodePolicy<entry_in_memory, TypeTag::Array>::remove_child(Node::iterator const&) { NOT_IMPLEMENTED; }
template <>
void NodePolicy<entry_in_memory, TypeTag::Array>::remove_children(Node::range const&) { NOT_IMPLEMENTED; }
template <>
Node::iterator NodePolicy<entry_in_memory, TypeTag::Array>::begin()
{
    NOT_IMPLEMENTED;
    return Node::iterator{};
}
template <>
Node::iterator NodePolicy<entry_in_memory, TypeTag::Array>::end()
{
    NOT_IMPLEMENTED;
    return Node::iterator{};
}
template <>
Node::const_iterator NodePolicy<entry_in_memory, TypeTag::Array>::cbegin() const
{
    NOT_IMPLEMENTED;
    return Node::const_iterator{};
}
template <>
Node::const_iterator NodePolicy<entry_in_memory, TypeTag::Array>::cend() const
{
    NOT_IMPLEMENTED;
    return Node::const_iterator{};
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Array>::push_back(const std::shared_ptr<Node>& p)
{
    std::shared_ptr<Node> e = p != nullptr ? p : convert_to(TypeTag::Null);

    try
    {
        std::get<array_t>(*entry()).push_back(e);
    }
    catch (const std::bad_variant_access&)
    {
        entry()->emplace<array_t>({e});
    }
    return e;
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Array>::push_back(Node&&)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Array>::push_back(const Node&)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Node::range NodePolicy<entry_in_memory, TypeTag::Array>::push_back(const Node::iterator& b, const Node::iterator& e)
{
    NOT_IMPLEMENTED;
    return Node::range{};
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Array>::at(int idx)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Node> NodePolicy<entry_in_memory, TypeTag::Array>::at(int idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

//---------------------------------------------------------------------------------------------------
// Table
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Table>::as_interface(TypeTag tag)
{
    return convert_to(tag);
}

template <>
size_t NodePolicy<entry_in_memory, TypeTag::Table>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Node::range NodePolicy<entry_in_memory, TypeTag::Table>::children()
{
    NOT_IMPLEMENTED;
    return Node::range{};
}
template <>
Node::const_range NodePolicy<entry_in_memory, TypeTag::Table>::children() const
{
    NOT_IMPLEMENTED;
    return Node::const_range{};
}
template <>
void NodePolicy<entry_in_memory, TypeTag::Table>::clear_children()
{
    NOT_IMPLEMENTED;
}
template <>
void NodePolicy<entry_in_memory, TypeTag::Table>::remove_child(Node::iterator const&)
{
    NOT_IMPLEMENTED;
}
template <>
void NodePolicy<entry_in_memory, TypeTag::Table>::remove_children(Node::range const&)
{
    NOT_IMPLEMENTED;
}
template <>
Node::iterator NodePolicy<entry_in_memory, TypeTag::Table>::begin()
{
    NOT_IMPLEMENTED;
    return Node::iterator{};
}
template <>
Node::iterator NodePolicy<entry_in_memory, TypeTag::Table>::end()
{
    NOT_IMPLEMENTED;
    return Node::iterator{};
}
template <>
Node::const_iterator NodePolicy<entry_in_memory, TypeTag::Table>::cbegin() const
{
    NOT_IMPLEMENTED;
    return Node::const_iterator{};
}
template <>
Node::const_iterator NodePolicy<entry_in_memory, TypeTag::Table>::cend() const
{
    NOT_IMPLEMENTED;
    return Node::const_iterator{};
}
template <>
Node::const_range_kv NodePolicy<entry_in_memory, TypeTag::Table>::items() const
{
    NOT_IMPLEMENTED;
    return Node::const_range_kv{};
}
template <>
Node::range_kv NodePolicy<entry_in_memory, TypeTag::Table>::items()
{
    NOT_IMPLEMENTED;
    return Node::range_kv{};
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Table>::insert(const std::string& k, std::shared_ptr<Node> const& node)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Node::range_kv NodePolicy<entry_in_memory, TypeTag::Table>::insert(const Node::iterator_kv& b, const Node::iterator_kv& e)
{
    NOT_IMPLEMENTED;
    return Node::range_kv{};
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Table>::at(const std::string& key)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Node> NodePolicy<entry_in_memory, TypeTag::Table>::at(const std::string& idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<Node> NodePolicy<entry_in_memory, TypeTag::Table>::find_child(const std::string&)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Node> NodePolicy<entry_in_memory, TypeTag::Table>::find_child(const std::string&) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

std::shared_ptr<Node> create_node(const std::string& str)
{
    return std::dynamic_pointer_cast<Node>(std::make_shared<NodeImplement<entry_in_memory, TypeTag::Null>>());
}
} // namespace sp
  // namespace sp
  // {
  // //----------------------------------------------------------------------------------------------------------
  // // Node
  // //----------------------------------------------------------------------------------------------------------

// Node::Node() : m_parent_(nullptr), m_entry_(create_entry()) {}

// Node::Node(Node* parent, NodeInterface* entry)
//     : m_parent_(parent),
//       m_entry_(entry != nullptr ? entry : create_entry()) {}

// Node::~Node() {}

// Node::Node(this_type const& other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_->copy()) {}

// Node::Node(this_type&& other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_.release())
// {
//     other.m_entry_.reset(other.m_entry_->create());
// }

// void Node::swap(this_type& other)
// {
//     std::swap(m_parent_, other.m_parent_);
//     std::swap(m_entry_, other.m_entry_);
// }

// Node& Node::operator=(this_type const& other)
// {
//     Node(other).swap(*this);
//     return *this;
// }

// Node* Node::copy() const { return new Node(m_parent_, m_entry_->copy()); }

// TypeTag Node::type_tag() const { return m_entry_ == nullptr ? TypeTag::Null : m_entry_->type_tag(); }

// bool Node::empty() const { return m_entry_ == nullptr; }

// bool Node::is_null() const { return type_tag() == TypeTag::Null; }

// bool Node::is_scalar() const { return type_tag() == TypeTag::Scalar; }

// bool Node::is_block() const { return type_tag() == TypeTag::Block; }

// bool Node::is_array() const { return type_tag() == TypeTag::Array; }

// bool Node::is_table() const { return type_tag() == TypeTag::Table; }

// bool Node::is_root() const { return m_parent_ == nullptr; }

// bool Node::is_leaf() const { return !(is_table() || is_array()); }

// size_t Node::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

// //----------------------------------------------------------------------------------------------------------
// // Node: Common
// //----------------------------------------------------------------------------------------------------------
// Node& Node::parent() const { return *m_parent_; }

// void Node::resolve() { m_entry_->resolve(); }

// Node* Node::create_child() { return new Node(this, m_entry_->create()); };

// Node& Node::as_scalar()
// {
//     m_entry_->as_scalar();
//     return *this;
// }

// Node& Node::as_block()
// {
//     m_entry_->as_block();
//     return *this;
// }

// Node& Node::as_array()
// {
//     m_entry_->as_array();
//     return *this;
// }

// Node& Node::as_table()
// {
//     m_entry_->as_table();
//     return *this;
// }

// //----------------------------------------------------------------------------------------------------------
// // as leaf node,
// //----------------------------------------------------------------------------------------------------------
// // function level 0

// std::any Node::get_scalar() const { return m_entry_->get_scalar(); }

// void Node::set_scalar(const std::any& v) { m_entry_->set_scalar(v); }

// std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>
// Node::get_raw_block() const
// {
//     return m_entry_->get_raw_block();
// }

// void Node::set_raw_block(const std::shared_ptr<void>& data /*data pointer*/,
//                          const std::type_info& t /*element type*/,
//                          const std::vector<size_t>& dims /*dimensions*/)
// {
//     m_entry_->set_raw_block(data, t, dims);
// }

// //----------------------------------------------------------------------------------------------------------
// // as tree node,
// //----------------------------------------------------------------------------------------------------------
// // function level 0
// size_t Node::size() const { return m_entry_->size(); }

// Node::range Node::children() { return m_entry_->children(); }

// // Node::const_range Node::children() const { return m_entry_->children(); }

// void Node::clear_children() { m_entry_->clear_children(); }

// void Node::remove_child(const iterator& it) { m_entry_->remove_child(it); }

// void Node::remove_children(const range& r) { m_entry_->remove_children(r); }

// Node::iterator Node::begin() { return m_entry_->begin(); }

// Node::iterator Node::end() { return m_entry_->end(); }

// Node::const_iterator Node::cbegin() const { return m_entry_->cbegin(); }

// Node::const_iterator Node::cend() const { return m_entry_->cend(); }

// // as table

// Node& Node::insert(const std::string& k, const std::shared_ptr<Node>& v)
// {
//     return *m_entry_->insert(k, v);
// }

// Node& Node::insert(const std::string& k, const Node& n)
// {
//     return *m_entry_->insert(k, std::make_shared<Node>(n));
// }

// Node& Node::insert(const std::string& k, Node&& n)
// {
//     return *m_entry_->insert(k, std::make_shared<Node>(std::move(n)));
// }

// Node::range_kv Node::insert(const iterator_kv& b, const iterator_kv& e)
// {
//     return m_entry_->insert(b, e);
// }

// Node::range_kv Node::insert(const range_kv& r)
// {
//     return m_entry_->insert(r.begin(), r.end());
// }

// // Node::const_range_kv Node::items() const
// // {
// //     return m_entry_->items();
// // }

// Node::range_kv Node::items()
// {
//     return m_entry_->items();
// }

// Node::iterator Node::find_child(const std::string& k)
// {
//     return iterator(m_entry_->find_child(k).get());
// }

// Node& Node::at(const std::string& k)
// {
//     return *m_entry_->at(k);
// }

// const Node& Node::at(const std::string& k) const
// {
//     return *m_entry_->at(k);
// }

// Node& Node::operator[](const std::string& path) { return at(path); }

// const Node& Node::operator[](const std::string& path) const { return at(path); }

// // as array

// Node& Node::push_back() { return *m_entry_->push_back(); }

// Node& Node::push_back(const std::shared_ptr<Node>& n) { return *m_entry_->push_back(n); }

// Node& Node::push_back(const Node& n) { return *m_entry_->push_back(n); }

// Node& Node::push_back(Node&& n) { return *m_entry_->push_back(std::move(n)); }

// Node::range Node::push_back(const range& r) { return m_entry_->push_back(r.begin(), r.end()); }

// Node::range Node::push_back(const iterator& b, const iterator& e) { return m_entry_->push_back(b, e); }

// Node::iterator Node::find_child(int idx) { return iterator(m_entry_->find_child(idx).get()); }

// Node& Node::at(int idx) { return *m_entry_->find_child(idx); }

// const Node& Node::at(int idx) const { return *m_entry_->find_child(idx); }

// Node& Node::operator[](size_t idx) { return at(idx); }

// const Node& Node::operator[](size_t idx) const { return at(idx); }

// // as Table
// // Node& Node::insert(std::string const&, std::shared_ptr<Node> const&);

// // Node& Node::operator[](std::string const& path) { return as_table()[path]; }

// // const Node& Node::operator[](std::string const& path) const { return as_table()[path]; }

// // // as Array
// // Node& Node::push_back(std::shared_ptr<Node> const&);

// // Node& Node::operator[](size_t idx) { return as_array()[idx]; }

// // const Node& Node::operator[](size_t idx) const { return as_array()[idx]; }
// //----------------------------------------------------------------------------------------------------------
// // function level 1
// Node::range Node::select(XPath const& path)
// {
//     NOT_IMPLEMENTED;
//     return Node::range();
// }

// Node::const_range Node::select(XPath const& path) const
// {
//     NOT_IMPLEMENTED;
//     return Node::const_range();
// }

// Node::iterator Node::select_one(XPath const& path)
// {
//     NOT_IMPLEMENTED;
//     return Node::iterator();
// }

// Node::const_iterator Node::select_one(XPath const& path) const
// {
//     NOT_IMPLEMENTED;
//     return Node::const_iterator();
// }

// } // namespace sp