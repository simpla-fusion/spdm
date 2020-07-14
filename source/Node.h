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
enum TypeTag
{
    Null = 0000, // value is invalid
    Scalar,
    Block,
    Tree,
    Array, // as JSON array
    Table  // key-value, C++ map or JSON object

};
class XPath;
class Attributes;
class EntryInterface;

class Node : public std::enable_shared_from_this<Node>
{
public:
    typedef Node this_type;
    typedef Iterator<Node> iterator;
    typedef Iterator<const Node> const_iterator;
    typedef Range<iterator> range;
    typedef Range<const_iterator> const_range;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<Node>>> iterator_kv;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<const Node>>> const_iterator_kv;
    typedef Range<iterator_kv> range_kv;
    typedef Range<const_iterator_kv> const_range_kv;

    Node();
    virtual ~Node();

    Node(this_type const& other);
    Node(this_type&& other);
    void swap(this_type& other);
    this_type& operator=(this_type const& other);

    Node* copy() const;

    TypeTag type_tag() const;

    bool empty() const;

    bool is_null() const;
    bool is_scalar() const;

    bool is_block() const;

    bool is_array() const;
    bool is_table() const;

    bool is_root() const; // if  parent is null then true else false
    bool is_leaf() const; // if type in [Null,Scalar,Block]

    //----------------------------------------------------------------------------------------------------------
    // Attribute
    //----------------------------------------------------------------------------------------------------------
    void resolve(); // resolve reference or operator in object

    Node& parent() const; // return parent node

    Node* create_child();

    Attributes& attributes();

    const Attributes& attributes() const;

    this_type& as_scalar();

    this_type& as_block();

    this_type& as_array();

    this_type& as_table();

    //----------------------------------------------------------------------------------------------------------
    // Node
    //----------------------------------------------------------------------------------------------------------

    std::any get_scalar() const; // get value , if value is invalid then throw exception

    void set_scalar(const std::any&);

    template <typename U, typename V>
    void set_value(V const& v) { set_scalar(std::make_any<U>(v)); }

    template <typename U>
    U get_value() const { return std::any_cast<U>(get_scalar()); }

    std::tuple<std::shared_ptr<void>, std::type_info const&, std::vector<size_t>> get_raw_block() const; // get block

    void set_raw_block(const std::shared_ptr<void>& /*data pointer*/, const std::type_info& /*element type*/, const std::vector<size_t>& /*dimensions*/); // set block

    template <typename V, typename... Args>
    void set_block(std::shared_ptr<V> const& d, Args... args)
    {
        set_raw_block(std::reinterpret_pointer_cast<void>(d),
                      typeid(V), std::vector<size_t>{std::forward<Args>(args)...});
    }

    template <typename V, typename... Args>
    std::tuple<std::shared_ptr<V>, std::type_info const&, std::vector<size_t>> get_block() const
    {
        auto blk = get_raw_block();
        return std::make_tuple(std::reinterpret_pointer_cast<void>(std::get<0>(blk)),
                               std::get<1>(blk), std::get<2>(blk));
    }
    //----------------------------------------------------------------------------------------------------------
    // as tree node,  need node.type = List || Object
    //----------------------------------------------------------------------------------------------------------
    // function level 0

    size_t size() const;

    range children(); // reutrn list of children

    const_range children() const; // reutrn list of children

    void clear_children();

    void remove_child(const iterator&);

    void remove_children(const range&);

    iterator begin();

    iterator end();

    const_iterator cbegin() const;

    const_iterator cend() const;

    // as table

    Node& insert(const std::string&, const std::shared_ptr<Node>&);

    Node& insert(const std::string&, const Node&);

    Node& insert(const std::string&, Node&&);

    range_kv insert(const iterator_kv& b, const iterator_kv& e);

    range_kv insert(const range_kv& r);

    template <typename TI0, typename TI1>
    auto insert(TI0 const& b, TI1 const& e) { return insert(iterator_kv(b), iterator_kv(e)); }

    const_range_kv items() const;

    range_kv items();

    iterator find_child(const std::string&);

    Node& at(const std::string&);

    const Node& at(const std::string&) const;

    Node& operator[](const std::string& path);

    const Node& operator[](const std::string& path) const;

    // as array

    Node& push_back();

    Node& push_back(const std::shared_ptr<Node>&);

    Node& push_back(const Node&);

    Node& push_back(Node&&);

    range push_back(const range&);

    range push_back(const iterator&, const iterator&);

    iterator find_child(int idx);

    Node& at(int idx);

    const Node& at(int idx) const;

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
    Node(Node* parent, EntryInterface* entry);

    Node* m_parent_;
    std::unique_ptr<EntryInterface> m_entry_;
};

} // namespace sp
std::ostream& operator<<(std::ostream& os, sp::Node const& d);

#endif // SP_NODE_H_