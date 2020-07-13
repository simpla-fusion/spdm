#ifndef SP_NODE_H_
#define SP_NODE_H_

#include "Range.h"
#include <algorithm>
#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
namespace sp
{

class XPath;
class Attributes;
class Entry;
class Node;

class Entry;
class ScalarEntry;
class TensorEntry;
class TreeEntry;
class ArrayEntry; // as JSON array
class TableEntry; // key-value, C++ map or JSON object

class Node : public std::enable_shared_from_this<Node>
{
public:
    enum TypeTag
    {
        Null = 0b0000, // value is invalid
        Scalar = 0b0010,
        Tensor = 0b0110,
        Array = 0b0011,
        Table = 0b0111
    };
    typedef Node this_type;
    typedef Iterator<Node> iterator;
    typedef Iterator<const Node> const_iterator;
    typedef Range<iterator> range;
    typedef Range<const_iterator> const_range;

    Node(Node* parent = nullptr, Entry* entry = nullptr);
    virtual ~Node();

    Node(this_type const& other);
    Node(this_type&& other);
    void swap(this_type& other);
    this_type& operator=(this_type const& other);

    static Node* create(TypeTag t, Node* parent = nullptr, std::string const& backend = "");

    Node* copy() const;

    TypeTag type_tag() const;

    bool empty() const;

    bool is_null() const;
    bool is_scalar() const;
    bool is_tensor() const;
    bool is_array() const;
    bool is_table() const;

    bool is_root() const; // if  parent is null then true else false
    bool is_leaf() const; // if type in [Null,Scalar,Block]

    //----------------------------------------------------------------------------------------------------------
    // Attribute
    //----------------------------------------------------------------------------------------------------------
    Attributes& attributes();

    const Attributes& attributes() const;

    //----------------------------------------------------------------------------------------------------------
    // Node
    //----------------------------------------------------------------------------------------------------------

    Node& parent() const; // return parent node

    void resolve(); // resolve reference or operator in object

    Node* create_child(TypeTag) const;

    ScalarEntry& as_scalar();

    const ScalarEntry& as_scalar() const;

    TensorEntry& as_tensor();

    const TensorEntry& as_tensor() const;

    TreeEntry& as_tree();

    const TreeEntry& as_tree() const;

    ArrayEntry& as_array();

    const ArrayEntry& as_array() const;

    TableEntry& as_table();

    const TableEntry& as_table() const;

    //----------------------------------------------------------------------------------------------------------
    // as tree node,  need node.type = List || Object
    //----------------------------------------------------------------------------------------------------------
    // function level 0

    size_t size() const;

    range children(); // reutrn list of children

    const_range children() const; // reutrn list of children

    void clear_children();

    void remove_child(iterator const&);

    void remove_children(range const&);

    std::shared_ptr<Node> find_child(std::string const&);

    std::shared_ptr<Node> find_child(int idx);

    iterator begin();

    iterator end();

    const_iterator cbegin() const;

    const_iterator cend() const;

    Node& push_back(std::shared_ptr<Node> const&);

    Node& insert(std::string const&, std::shared_ptr<Node> const&);

    Node& operator[](std::string const& path);

    const Node& operator[](std::string const& path) const;

    Node& operator[](size_t idx);

    const Node& operator[](size_t idx) const;

    //----------------------------------------------------------------------------------------------------------
    // function level 1

    const_range select(XPath const& path) const; // select from children

    range select(XPath const& path); // select from children

    iterator select_one(XPath const& path); // return refernce of the first selected child  , if fail then throw exception

    const_iterator select_one(XPath const& path) const; // return refernce of the first selected child , if fail then throw exception

    //----------------------------------------------------------------------------------------------------------
    // function level 2
    ptrdiff_t distance(const this_type& target) const; // lenght of short path to target

    size_t depth() const; // parent.depth +1

    size_t height() const; // max(leaf.height) +1

    const_iterator first_child() const; // return iterator of the first child;

    iterator first_child(); // return iterator of the first child;

    const_range slibings() const; // return slibings

    const_range ancestor() const; // return ancestor

    const_range descendants() const; // return descendants

    const_range leaves() const; // return leave nodes in traversal order

    const_range path(this_type const& target) const; // return the shortest path to target

private:
    Node* m_parent_;
    std::unique_ptr<Entry> m_entry_;
};

class Entry
{
public:
    Entry() = default;

    virtual ~Entry() = default;

    virtual Node::TypeTag type_tag() const { return Node::TypeTag::Null; }

    virtual Entry* create(Node::TypeTag t = Node::TypeTag::Null) = 0;

    virtual Entry* copy() const = 0;

    virtual Attributes& attributes() = 0;

    virtual const Attributes& attributes() const = 0;
};

class ScalarEntry : public Entry
{
public:
    ScalarEntry() = default;

    virtual ~ScalarEntry() = default;

    Node::TypeTag type_tag() const { return Node::Scalar; }

    virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

    virtual void set_scalar(std::any const&) = 0;

    virtual std::tuple<std::shared_ptr<char>, std::type_info const&, std::vector<size_t>> get_raw_block() const = 0; // get block

    virtual void set_raw_block(std::shared_ptr<char> const& /*data pointer*/, std::type_info const& /*element type*/, std::vector<size_t> const& /*dimensions*/) = 0; // set block

    template <typename U, typename V>
    void set_value(V const& v) { set_scalar(std::make_any<U>(v)); }

    template <typename U>
    U get_value() const { return std::any_cast<U>(get_scalar()); }
};

class TensorEntry : public Entry
{

public:
    TensorEntry() = default;

    virtual ~TensorEntry() = default;

    Node::TypeTag type_tag() const { return Node::Tensor; }

    // template <typename V, typename... Args>
    // void set_block(std::shared_ptr<V> const& d, Args... args) { set_raw_block(std::reinterpret_pointer_cast<char>(d), typeid(V), std::vector<size_t>{std::forward<Args>(args)...}); }

    // template <typename V, typename... Args>
    // std::tuple<std::shared_ptr<V>, std::type_info const&, std::vector<size_t>> const get_block() const
    // {
    //     auto blk = get_raw_block();
    //     return std::make_tuple(std::reinterpret_pointer_cast<char>(std::get<0>(blk)), std::get<1>(blk), std::get<2>(blk));
    // }
};

class TreeEntry : public Entry
{

public:
    TreeEntry() = default;

    virtual ~TreeEntry() = default;

    virtual size_t size() const = 0;

    virtual Node::range children() = 0; // reutrn list of children

    virtual Node::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(Node::iterator const&) = 0;

    virtual void remove_children(Node::range const&) = 0;

    virtual Node::iterator begin() = 0;

    virtual Node::iterator end() = 0;

    virtual Node::const_iterator cbegin() const = 0;

    virtual Node::const_iterator cend() const = 0;
};

class ArrayEntry : public TreeEntry
{
public:
    ArrayEntry() = default;

    virtual ~ArrayEntry() = default;

    Node::TypeTag type_tag() const { return Node::Array; }

    virtual size_t size() const = 0;

    virtual Node::range children() = 0; // reutrn list of children

    virtual Node::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(Node::iterator const&) = 0;

    virtual void remove_children(Node::range const&) = 0;

    virtual Node::iterator begin() = 0;

    virtual Node::iterator end() = 0;

    virtual Node::const_iterator cbegin() const = 0;

    virtual Node::const_iterator cend() const = 0;

    virtual Node& push_back(const std::shared_ptr<Node>& p = nullptr) = 0;

    virtual Node push_back(Node&&) = 0;

    virtual Node push_back(const Node&) = 0;

    virtual Node::range push_back(const Iterator<std::shared_ptr<Node>>& b, const Iterator<std::shared_ptr<Node>>& e) = 0;

    // template <typename TI0, typename TI1>
    // Node::range push_back(TI0 const& b, TI1 const&e)
    // {
    //     return push_back(Iterator<std::shared_ptr<Node>>(b), Iterator<std::shared_ptr<Node>>(e));
    // }

    virtual Node& at(int idx) = 0;

    virtual const Node& at(int idx) const = 0;

    Node& operator[](size_t idx) { return at(idx); }

    const Node& operator[](size_t idx) const { return at(idx); }

    virtual std::shared_ptr<Node> find_child(size_t) = 0;

    virtual std::shared_ptr<const Node> find_child(size_t) const = 0;
};

class TableEntry : public TreeEntry
{
public:
    TableEntry() = default;

    virtual ~TableEntry() = default;

    Node::TypeTag type_tag() const { return Node::Table; }

    virtual size_t size() const = 0;

    virtual Node::range children() = 0; // reutrn list of children

    virtual Node::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(Node::iterator const&) = 0;

    virtual void remove_children(Node::range const&) = 0;

    virtual Node::iterator begin() = 0;

    virtual Node::iterator end() = 0;

    virtual Node::const_iterator cbegin() const = 0;

    virtual Node::const_iterator cend() const = 0;

    virtual Node& insert(std::string const& k, std::shared_ptr<Node> const& node) = 0; //

    virtual Node::range insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const& b, Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const& e) = 0;

    template <typename TI0, typename TI1>
    auto insert(TI0 const& b, TI1 const& e)
    {
        return insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>>(b),
                      Iterator<std::pair<const std::string, std::shared_ptr<Node>>>(e));
    }

    virtual Node& at(std::string const& idx) = 0;

    virtual const Node& at(std::string const& idx) const = 0;

    Node& operator[](std::string const& path) { return at(path); }

    const Node& operator[](std::string const& path) const { return at(path); }

    virtual std::shared_ptr<Node> find_child(std::string const&) = 0;

    virtual std::shared_ptr<const Node> find_child(std::string const&) const = 0;
};
} // namespace sp
std::ostream& operator<<(std::ostream& os, sp::Node const& d);

#endif // SP_NODE_H_