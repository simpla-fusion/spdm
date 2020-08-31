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
class NodeBackend;
class NodeObject;
class NodeArray;
class DataBlock;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Block, sp::db::DataBlock);
M_REGISITER_TYPE_TAG(Path, sp::db::Path);

M_REGISITER_TYPE_TAG(Object, sp::db::NodeObject);
M_REGISITER_TYPE_TAG(Array, sp::db::NodeArray);

namespace sp::db
{

class NodeObject
{
private:
    std::shared_ptr<NodeBackend> m_backend_;
    NodeBackend& backend();
    const NodeBackend& backend() const;

public:
    NodeObject();

    ~NodeObject();

    NodeObject(std::initializer_list<std::pair<std::string, Node>> init);

    NodeObject(const NodeObject&);

    NodeObject(NodeObject&&);

    template <typename... Args>
    NodeObject(Args&&... args) {}

    void swap(NodeObject& other) { std::swap(m_backend_, other.m_backend_); }

    NodeObject& operator=(const NodeObject& other)
    {
        NodeObject(other).swap(*this);
        return *this;
    }

    static NodeObject create(const NodeObject& opt);

    void load(const NodeObject&);

    void save(const NodeObject&) const;

    bool is_valid() const;

    bool empty() const;

    size_t size() const;

    void clear();

    void reset();

    Cursor<Node> children();

    Cursor<const Node> children() const;

    void for_each(std::function<void(const std::string&, const Node&)> const&) const;

    Node fetch(const NodeObject& data, const NodeObject& opt = {});

    Node fetch(const NodeObject& data, const NodeObject& opt = {}) const;

    void update(NodeObject, const NodeObject& opt = {});

    void set_value(const std::string& name, Node v);

    Node get_value(const std::string& name) const;
};

class NodeArray
{
    std::vector<Node> m_container_;

public:
    NodeArray() = default;

    ~NodeArray() = default;

    template <typename IT>
    NodeArray(const IT& ib, const IT& ie) : m_container_(ib, ie) {}

    NodeArray(const NodeArray& other) : m_container_(other.m_container_) {}

    NodeArray(NodeArray&& other) : m_container_(std::move(other.m_container_)) {}

    void swap(NodeArray& other) { m_container_.swap(other.m_container_); }

    NodeArray& operator=(const NodeArray& other)
    {
        NodeArray(other).swap(*this);
        return *this;
    }

    void clear();

    size_t size() const;

    Cursor<Node> children();

    Cursor<const Node> children() const;

    void for_each(std::function<void(int, Node&)> const&);

    void for_each(std::function<void(int, const Node&)> const&) const;

    Node slice(int start, int stop, int step);

    Node slice(int start, int stop, int step) const;

    void resize(std::size_t num);

    Node& insert(int idx, Node);

    Node& at(int idx);

    const Node& at(int idx) const;

    Node& push_back(Node v);

    Node pop_back();
};

class Node
{

public:
    typedef std::variant<std::nullptr_t,
                         NodeObject,                         //Object
                         NodeArray,                          //Array
                         DataBlock,                          //Block
                         Path,                               //Path
                         bool,                               //Boolean,
                         int,                                //Integer,
                         long,                               //Long,
                         float,                              //Float,
                         double,                             //Double,
                         std::string,                        //String,
                         std::array<int, 3>,                 //IntVec3,
                         std::array<long, 3>,                //LongVec3,
                         std::array<float, 3>,               //FloatVec3,
                         std::array<double, 3>,              //DoubleVec3,
                         std::complex<double>,               //Complex,
                         std::array<std::complex<double>, 3> //ComplexVec3,
                         >
        value_type;

    typedef traits::type_tags<value_type> tags;

private:
    value_type m_value_;

public:
    Node() = default;

    template <typename... Args,
              std::enable_if_t<std::is_constructible<value_type, Args...>::value, int> = 0>
    Node(Args&&... args) : m_value_(std::forward<Args>(args)...) {}

    Node(char const* c) : m_value_(std::string(c)) {}

    Node(std::initializer_list<Node> init);

    Node(const Node& other) : m_value_(other.m_value_) {}

    Node(Node&& other) : m_value_(std::move(other.m_value_)) {}

    ~Node() = default;

    value_type& value() { return m_value_; }

    const value_type& value() const { return m_value_; }

    void swap(Node& other) { std::swap(m_value_, other.m_value_); }

    NodeArray& as_array() { return std::get<tags::Array>(m_value_); }

    const NodeArray& as_array() const { return std::get<tags::Array>(m_value_); }

    NodeObject& as_object() { return std::get<tags::Object>(m_value_); }

    const NodeObject& as_object() const { return std::get<tags::Object>(m_value_); }
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