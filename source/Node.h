#ifndef SP_NODE_H_
#define SP_NODE_H_
#include "HierarchicalTree.h"
#include "utility/Logger.h"
#include <any>
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <variant>
#include <vector>
namespace sp
{
class Entry;
class Node;
template <>
struct cursor_traits<Node>
{
    typedef Node value_type;
    typedef Node reference;
    typedef std::shared_ptr<Node> pointer;
    typedef ptrdiff_t difference_type;
};
template <typename TNode>
class NodeObjectHolder
{
    TNode* m_self_;

public:
    typedef TNode node_type;
    typedef Cursor<TNode> cursor;

    NodeObjectHolder(TNode* self = nullptr) : m_self_(self){};

    ~NodeObjectHolder() = default;

    void swap(NodeObjectHolder& other) { std::swap(m_entry_, other.m_entry_); }

    size_t size() const { return 0; }

    size_t count(const std::string& key) const { return 0; }

    void clear() {}

    template <typename... Args>
    std::pair<cursor, bool> try_emplace(const std::string& key, Args&&... args) { return std::make_pair(cursor(), false); }

    cursor insert(const std::string& path) {}

    void erase(const std::string& key) {}

    cursor find(const std::string& key) const { return cursor{}; }

    cursor find(const std::string& key) { return cursor{}; }

private:
    std::shared_ptr<Entry> m_entry_;
};

template <typename TNode>
class NodeArrayHolder
{
    TNode* m_self_;

public:
    typedef TNode node_type;
    typedef Cursor<TNode> cursor;

    NodeArrayHolder(node_type* self = nullptr) : m_self_(self){};
    NodeArrayHolder(NodeArrayHolder&&) {}
    NodeArrayHolder(const NodeArrayHolder&) {}
    ~NodeArrayHolder() = default;

    void swap(NodeArrayHolder& other) { std::swap(m_entry_, other.m_entry_); }

    NodeArrayHolder& operator=(NodeArrayHolder const& other)
    {
        NodeArrayHolder(other).swap(*this);
        return *this;
    }

    size_t size() const { return 0; }

    void resize(size_t s){};

    void clear() {}

    template <typename... Args>
    cursor emplace_back(Args&&... args)
    {
        // m_entry_->emplace_back(std::forward<Args>(args)...);
        // return m_entry_->back();
        return cursor{};
    }

    void pop_back() {}

    node_type at(int idx) { return node_type{}; }

    node_type at(int idx) const { return node_type{}; }

private:
    std::shared_ptr<Entry> m_entry_;
};

class Node : public HierarchicalNode<Node, NodeObjectHolder, NodeArrayHolder>
{
public:
    typedef HierarchicalNode<Node, NodeObjectHolder, NodeArrayHolder> base_type;
    typedef Node this_type;

    Node(Node* parent, const std::string& name) : base_type(parent, name) {}

    Node(const std::string& backend) : base_type(nullptr, "", std::integral_constant<int, DataType::Object>(), backend) {}

    ~Node() = default;

    this_type& operator=(Node const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    // attributes

    bool has_attribute(const std::string& name) const { return has_a("@" + name); }

    void remove_attribute(const std::string& name) { erase("@" + name); }

    template <typename V>
    auto get_attribute(const std::string& name) { return this->operator[]("@" + name).get_value<V>(); };

    template <typename V, typename U>
    void set_attribute(const std::string& name, const U& value) { this->operator[]("@" + name).set_value<V>(value); }

    // private:
    //     this_type* m_parent_;
    //     std::string m_name_;
    //     data_type& fetch()
    //     {
    //         if (m_data_.index() == NULL_TAG && m_parent_ != nullptr && m_name_ != "")
    //         {
    //             m_parent_->fetch(m_name_).swap(m_data_);
    //         }

    //         return m_data_;
    //     }

    //     const data_type& fetch() const { return const_cast<this_type*>(this)->fetch(); }

    //     data_type fetch(const std::string& path) { return get_r(path).m_data_; }

    //     const data_type& fetch(const std::string& path) const { return get_r(path).m_data_; }
};

std::ostream& operator<<(std::ostream& os, Node const& Node);

} // namespace sp

#endif // SP_NODE_H_
