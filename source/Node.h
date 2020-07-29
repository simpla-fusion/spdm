#ifndef SP_NODE_H_
#define SP_NODE_H_
#include "Entry.h"
#include "utility/Cursor.h"
#include "utility/HierarchicalTree.h"
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
class Node;
template <>
struct node_traits<Node>
{
    typedef Node node_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;
    typedef node_type& reference;
    typedef node_type* pointer;
    typedef Entry object_container;
    typedef Entry array_container;
};

class Node : public hierarchical_tree_wrapper<Node, Entry::type_union, Entry::type_tag>::type
{
public:
    typedef Node this_type;

    typedef typename hierarchical_tree_wrapper<Node, Entry::type_union, Entry::type_tag>::type base_type;

    typedef typename hierarchical_tree_wrapper<Node, Entry::type_union, Entry::type_tag>::type tag;

    Node(const std::string& backend);

    Node(Node* parent = nullptr, const std::string& name = "");

    Node(const this_type& other) : base_type(other){};

    Node(this_type&& other) : base_type(std::move(other)){};

    ~Node() = default;

    void swap(this_type& other)
    {
        base_type::swap(other);
    }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    // attributes

    bool has_attribute(const std::string& name) const { return count("@" + name) > 0; }

    void erase_attribute(const std::string& name) { erase("@" + name); }

    template <typename V>
    auto get_attribute(const std::string& name) { return find("@" + name)->get_value<V>(); };

    template <typename V, typename U>
    void set_attribute(const std::string& name, const U& value) { insert("@" + name, this)->template set_value<V>(value); }
};

std::ostream& operator<<(std::ostream& os, Node const& Node);

} // namespace sp

#endif // SP_NODE_H_
