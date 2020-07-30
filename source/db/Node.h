#ifndef SP_NODE_H_
#define SP_NODE_H_
#include "../utility/Logger.h"
#include "Cursor.h"
#include "Entry.h"
#include "HierarchicalTree.h"
#include <any>
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <variant>
#include <vector>
namespace sp::db
{
class Node;

template <>
struct cursor_traits<Node>
{
    typedef Node value_type;
    typedef Node reference;
    typedef std::shared_ptr<Node> pointer;
    typedef ptrdiff_t difference_type;
};
template <>
struct cursor_traits<const Node>
{
    typedef const Node value_type;
    typedef const Node reference;
    typedef std::shared_ptr<const Node> pointer;
    typedef ptrdiff_t difference_type;
};
 

class Node : public entry_wrapper<Node>
{
public:
    typedef Node this_type;

    typedef entry_wrapper<Node> base_type;

    using typename base_type::element;

    using typename base_type::type_tags;

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
    void set_attribute(const std::string& name, const U& value) { insert("@" + name)->template set_value<V>(value); }
};

std::ostream& operator<<(std::ostream& os, Node const& Node);

} // namespace sp::db

#endif // SP_NODE_H_
