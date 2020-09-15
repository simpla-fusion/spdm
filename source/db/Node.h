#ifndef SPDB_NODE_H_
#define SPDB_NODE_H_
#include "../utility/TypeTraits.h"
#include "Cursor.h"
#include "DataBlock.h"
#include "XPath.h"
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
namespace sp::db
{
class Node;
class NodeObject;
class NodeArray;
class DataBlock;
class Unknown
{
};
} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::NodeObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::NodeArray>);
M_REGISITER_TYPE_TAG(Block, sp::db::DataBlock);
M_REGISITER_TYPE_TAG(Path, sp::db::Path);
M_REGISITER_TYPE_TAG(Unknown, sp::db::Unknown);

namespace sp::db
{
#define DEFAULT_SCHEMA_TAG "_schema"

typedef std::variant<std::nullptr_t,
                     std::shared_ptr<NodeObject>, //Object
                     std::shared_ptr<NodeArray>,  //Array
                     DataBlock,                   //Block
                     Path,                        //Path
                     std::string,                 //String,
                     bool,                        //Boolean,
                     int,                         //Integer,
                     double,                      //Double,
                     std::complex<double>,        //Complex,
                                                  //std::array<int, 3>,                 //IntVec3,
                                                  //std::array<long, 3>,                //LongVec3,
                                                  //std::array<float, 3>,               //FloatVec3,
                                                  //std::array<double, 3>,              //DoubleVec3,
                                                  //std::array<std::complex<double>, 3> //ComplexVec3,
                     Unknown                      //Unknown
                     >
    node_value_type;

class Node
{
public:
    typedef node_value_type value_type;

    typedef traits::type_tags<value_type> tags;

    Node() = default;

    ~Node() = default;

    template <typename... Args,
              std::enable_if_t<sizeof...(Args) != 0 && std::is_constructible<value_type, Args...>::value, int> = 0>
    Node(Args&&... args) : m_value_(std::forward<Args>(args)...) {}

    Node(char const* c);

    Node(const std::initializer_list<Node>& init);

    Node(Node& other);

    Node(const Node& other);

    Node(Node&& other);

    Node& operator=(const Node& other)
    {
        Node(other).swap(*this);
        return *this;
    }

    bool operator==(const Node& other) const
    {
        NOT_IMPLEMENTED;
        return false;
    }

    void swap(Node& other);

    size_t type() const;

    bool is_null() const { return type() == tags::Null; }

    void clear();

    NodeArray& as_array();

    const NodeArray& as_array() const;

    NodeObject& as_object();

    const NodeObject& as_object() const;

    Path as_path() const;

    void value(value_type v) { m_value_.swap(v); }

    value_type& value() { return m_value_; }

    const value_type& value() const { return m_value_; }

    template <typename V, typename... Others>
    void set_value(Others&&... others) { m_value_.emplace<V>(std::forward<Others>(others)...); }

    template <int IDX, typename... Others>
    void set_value(Others&&... others) { m_value_.emplace<IDX>(std::forward<Others>(others)...); }

    template <typename V, typename... Args>
    V get_value(Args&&... args) const
    {
        V res;
        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<tags::Null, value_type>& ele) { res = V{std::forward<Args>(args)...}; },
                [&](auto&& v) { res = traits::convert<V>(v); }},
            m_value_);
        return res;
    }

    template <int IDX, typename... Args>
    auto get_value(Args&&... args) const
    {
        typedef std::variant_alternative_t<IDX, value_type> res_type;

        res_type res;

        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<tags::Null, value_type>) {
                    res = res_type{std::forward<Args>(args)...};
                },
                [&](auto&& v) { res = traits::convert<res_type>(v); }},
            m_value_);
        return res;
    }

    template <int IDX>
    decltype(auto) as() const { return std::get<IDX>(m_value_); }

    template <int IDX>
    decltype(auto) as() { return std::get<IDX>(m_value_); }

    template <typename Visitor>
    auto visit(Visitor const& visitor) { return std::visit(visitor, m_value_); }

    template <typename Visitor>
    auto visit(Visitor const& visitor) const { return std::visit(visitor, m_value_); }

private:
    value_type m_value_;
}; // namespace sp::db

class NodeObject : public std::enable_shared_from_this<NodeObject>
{

public:
    NodeObject() = default;

    virtual ~NodeObject() = default;

    NodeObject(const NodeObject&) = delete;

    NodeObject(NodeObject&&) = delete;

    static std::shared_ptr<NodeObject> create(const Node& opt = {});

    virtual std::shared_ptr<NodeObject> copy() const = 0;

    virtual void load(const Node&) = 0;

    virtual void save(const Node&) const = 0;

    virtual bool is_same(const NodeObject&) const = 0;

    virtual bool empty() const = 0;

    virtual void clear() = 0;

    virtual Cursor<Node> children() = 0;

    virtual Cursor<const Node> children() const = 0;

    virtual void for_each(std::function<void(const Node&, Node&)> const&) = 0;

    virtual void for_each(std::function<void(const Node&, const Node&)> const&) const = 0;

    //----------------

    virtual Node update(const Path&, const Node& = {}) = 0;

    virtual const Node fetch(const Path&, const Node& projection = {}) const = 0;
    //----------------

    virtual bool contain(const std::string&) const = 0;

    virtual void update_child(const std::string&, const Node&) = 0;

    virtual Node insert_child(const std::string&, const Node&) = 0;

    virtual Node find_child(const std::string&) const = 0;

    virtual void remove_child(const std::string&) = 0;
};

class NodeArray : public std::enable_shared_from_this<NodeArray>
{
    std::vector<Node> m_container_;

public:
    NodeArray() = default;

    virtual ~NodeArray() = default;

    NodeArray(const Node&);

    template <typename IT>
    NodeArray(const IT& ib, const IT& ie) : m_container_(ib, ie) {}

    NodeArray(const std::initializer_list<Node>& init);

    NodeArray(const NodeArray& other);

    NodeArray(NodeArray&& other);

    static std::shared_ptr<NodeArray> create(const Node& opt = {});

    void swap(NodeArray& other);

    NodeArray& operator=(const NodeArray& other);

    void clear();

    size_t size() const;

    Cursor<Node> children();

    Cursor<const Node> children() const;

    bool is_simple() const;

    int value_type() const;

    void for_each(std::function<void(const Node&, Node&)> const&);

    void for_each(std::function<void(const Node&, const Node&)> const&) const;

    Node slice(int start, int stop, int step);

    Node slice(int start, int stop, int step) const;

    void resize(std::size_t num);

    Node& insert(int idx, Node const&);

    Node& update(int idx, Node const&);

    Node& at(int idx);

    const Node& at(int idx) const;

    Node& push_back(Node const& v = {});

    Node pop_back();
};

std::ostream& operator<<(std::ostream& os, Node const& node);
std::ostream& operator<<(std::ostream& os, NodeObject const& node);
std::ostream& operator<<(std::ostream& os, NodeArray const& node);

namespace literals
{
using namespace std::complex_literals;
using namespace std::string_literals;
} // namespace literals
} // namespace sp::db

#endif //SP_NODE_H_