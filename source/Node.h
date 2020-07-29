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

template <typename V>
struct cursor_traits<V,
                     std::enable_if_t<
                         std::is_same_v<V, Node> ||
                         std::is_same_v<V, const Node>>>
{
    typedef Node value_type;
    typedef Node reference;
    typedef std::shared_ptr<Node> pointer;
    typedef ptrdiff_t difference_type;
};
template <>
struct node_traits<Node>
{
    typedef Node node_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;
    typedef typename cursor::reference reference;
    typedef typename cursor::pointer pointer;
    typedef Entry object_container;
    typedef Entry array_container;
};

class Node : public HierarchicalTree<
                 Node,
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
                 std::any>,
             public std::enable_shared_from_this<Node>
{
public:
    typedef Node this_type;

    typedef tree_type base_type;

    using typename tree_type::type_tags;

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
// template <>
// HierarchicalTreeObjectContainer<Node>;
// template <>
// HierarchicalTreeArrayContainer<Node>;
} // namespace sp

#endif // SP_NODE_H_
