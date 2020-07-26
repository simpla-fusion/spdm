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

class Node;

typedef HierarchicalTree<
    Node,
    HierarchicalTreeObjectPolicy,                                //Object
    HierarchicalTreeArrayPolicy,                                 //Array
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

    template <typename... Args>
    Node(Args&&... args)
        : base_type(nullptr, std::integral_constant<int, base_type::OBJECT_TAG>(), std::forward<Args>(args)...) {}

    ~Node() = default;
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
