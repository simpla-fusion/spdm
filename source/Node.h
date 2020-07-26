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
template <typename TNode>
class Cursor
{
public:
    TNode operator*() const { return TNode{}; }
    std::shared_ptr<TNode> operator->() const { return nullptr; }
    operator bool() const { return false; }
};

template <typename TNode>
class NodeObjectPolicy
{
    TNode* m_self_;

public:
    typedef TNode node_type;
    typedef Cursor<TNode> cursor;

    NodeObjectPolicy(TNode* self = nullptr) : m_self_(self){};
    NodeObjectPolicy(const std::string& backend) {}
    NodeObjectPolicy(NodeObjectPolicy&&) {}
    NodeObjectPolicy(const NodeObjectPolicy&) {}
    ~NodeObjectPolicy() = default;

    void swap(NodeObjectPolicy& other) { std::swap(m_entry_, other.m_entry_); }

    NodeObjectPolicy& operator=(NodeObjectPolicy const& other)
    {
        NodeObjectPolicy(other).swap(*this);
        return *this;
    }

    size_t size() const { return 0; }

    void clear() {}

    bool has_a(const std::string& key) const { return !!find(key); }

    node_type at(const std::string& key) const;

    node_type operator[](const std::string& key) { return node_type(m_self_, key); }

    node_type operator[](const std::string& key) const { return node_type(m_self_, key); }

    template <typename... Args>
    cursor try_emplace(const std::string& key, Args&&... args) { return cursor(); }

    cursor insert(const std::string& path) {}

    void erase(const std::string& key) {}

    // class iterator;

    cursor find(const std::string& key) const { return cursor{}; }

    cursor find(const std::string& key) { return cursor{}; }

    // template <typename... Args>
    // const iterator find(const std::string&, Args&&... args) const;

    // template <typename... Args>
    // int erase(Args&&... args);

private:
    std::shared_ptr<Entry> m_entry_;
};

template <typename TNode>
class NodeArrayPolicy
{
    TNode* m_self_;

public:
    typedef TNode node_type;
    typedef Cursor<TNode> cursor;

    NodeArrayPolicy(node_type* self = nullptr) : m_self_(self){};
    NodeArrayPolicy(NodeArrayPolicy&&) {}
    NodeArrayPolicy(const NodeArrayPolicy&) {}
    ~NodeArrayPolicy() = default;

    void swap(NodeArrayPolicy& other) { std::swap(m_entry_, other.m_entry_); }

    NodeArrayPolicy& operator=(NodeArrayPolicy const& other)
    {
        NodeArrayPolicy(other).swap(*this);
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

typedef HierarchicalTree<
    Node,
    NodeObjectPolicy,                                            //Object
    NodeArrayPolicy,                                             //Array
    std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
    std::string,                                                 //String,
    bool,                                                        //Boolean,
    int,                                                         //Integer,
    long,                                                        //Long,
    float,                                                       //Float,
    double,                                                      //Double,
    std::complex<double>,                                        //Complex,
    std::array<int, 3>,                                          //IntVec3,
    std::array<long, 3>,                                         //LongVec3,
    std::array<float, 3>,                                        //FloatVec3,
    std::array<double, 3>,                                       //DoubleVec3,
    std::array<std::complex<double>, 3>,                         //ComplexVec3,
    std::any                                                     //Other
    >
    tree_type;

class Node : public tree_type
{
public:
    typedef tree_type base_type;
    typedef Node this_type;

    enum DataType
    {
        Null,
        Object,
        Array,
        Block,
        String,
        Bool,
        Int,
        Long,
        Float,
        Double,
        Complex,
        IntVec3,
        LongVec3,
        FloatVec3,
        DoubleVec3,
        ComplexVec3,
        Other
    };

    using base_type::HierarchicalTree;

    using base_type::erase;
    using base_type::get_value;
    using base_type::set_value;
    using base_type::operator[];
    using base_type::has_a;

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

// class Node
// {
// private:
//     std::string m_path_;

// public:
//     typedef Node this_type;
//     typedef HierarchicalTree<EntryObject, EntryArray> base_type;

//     class cursor;

//     explicit Node(const std::string& uri = "");

//     explicit Node(const std::shared_ptr<Entry>& p, const std::string& prefix = "");

//     Node(const this_type&);

//     Node(this_type&&);

//     ~Node();

//     void swap(this_type&);

//     this_type& operator=(this_type const& other);

//     bool operator==(this_type const& other) const;

//     operator bool() const { return !is_null(); }

//     void resolve();

//     //

//     std::string path() const;

//     std::string name() const;

//     const Node& value() const;

//     Node& value();

//     // attributes

//     bool has_attribute(const std::string& name) const;

//     const Entry::element_t get_attribute_raw(const std::string& name) const;

//     void set_attribute_raw(const std::string& name, const Entry::element_t& value);

//     void remove_attribute(const std::string& name);

//     template <typename V>
//     const V get_attribute(const std::string& name)
//     {
//         return std::get<V>(get_attribute_raw(name));
//     };

//     template <typename V, typename U>
//     void set_attribute(const std::string& name, const U& value)
//     {
//         set_attribute_raw(name, Entry::element_t{V(value)});
//     }

//     std::map<std::string, Entry::element_t> attributes() const;

//     //----------------------------------------------------------------------------------
//     // level 0
//     //
//     // as leaf

//     // as Tree
//     // as container

//     Node* parent() const;

//     // as array

//     //-------------------------------------------------------------------
//     // level 1
//     // xpath

//     Node insert(const XPath&);

//     cursor find(const XPath&) const;

//     typedef std::function<bool(const Node&)> pred_fun;

//     cursor find(const pred_fun&) const;

//     int update(cursor&, const Node&);

//     int remove(cursor&);
// };

// class Node::cursor
// {
// public:
//     cursor();

//     cursor(Entry*);

//     cursor(const std::shared_ptr<Entry>&);

//     cursor(const cursor&);

//     cursor(cursor&&);

//     ~cursor() = default;

//     bool operator==(const cursor& other) const;

//     bool operator!=(const cursor& other) const;

//     operator bool() const { return !is_null(); }

//     bool is_null() const;

//     Node operator*();

//     std::unique_ptr<Node> operator->();

//     cursor& operator++();

//     cursor operator++(int);

// private:
//     std::shared_ptr<Entry> m_entry_;
// };

#endif // SP_NODE_H_
