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

class NodeObject;
class NodeArray;
class DataBlock;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::NodeObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::NodeArray>);
M_REGISITER_TYPE_TAG(Block, std::shared_ptr<sp::db::DataBlock>);

namespace sp::db
{

typedef std::variant<std::nullptr_t,
                     std::shared_ptr<NodeObject>,        //Object
                     std::shared_ptr<NodeArray>,         //Array
                     std::shared_ptr<DataBlock>,         //Block
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
    tree_node_type;

typedef traits::type_tags<tree_node_type> tree_node_tags;

class NodeObject : public std::enable_shared_from_this<NodeObject>
{

public:
    NodeObject() = default;
    virtual ~NodeObject() = default;
    NodeObject(const NodeObject&) = delete;
    NodeObject(NodeObject&&) = delete;

    static std::shared_ptr<NodeObject> create(const tree_node_type& opt = {});

    virtual void load(const tree_node_type&) { NOT_IMPLEMENTED; }

    virtual void save(const tree_node_type&) const { NOT_IMPLEMENTED; }

    virtual std::pair<std::shared_ptr<const NodeObject>, Path> full_path() const;

    virtual std::pair<std::shared_ptr<NodeObject>, Path> full_path();

    virtual std::unique_ptr<NodeObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;
    //-------------------------------------------------------------------------------------------------------------
    // as container

    virtual Cursor<tree_node_type> children() = 0;

    virtual Cursor<const tree_node_type> children() const = 0;

    // virtual void for_each(std::function<void(const std::string&, tree_node_type&)> const&) = 0;

    virtual void for_each(std::function<void(const std::string&, const tree_node_type&)> const&) const = 0;

    //------------------------------------------------------------------------------
    // fundamental operation ： CRUD
    /**
     *  Create 
     */
    virtual tree_node_type insert(Path p, tree_node_type) = 0;
    /**
     * Modify
     */
    virtual void update(Path p, tree_node_type) = 0;
    /**
     * Retrieve
     */
    virtual tree_node_type find(Path path = {}) const = 0;
    /**
     *  Delete 
     */
    virtual void remove(Path path = {}) = 0;

    //------------------------------------------------------------------------------
    // advanced extension functions
    virtual void merge(const NodeObject&);

    virtual void patch(const NodeObject&);

    virtual void update(const NodeObject&);

    virtual bool compare(const tree_node_type& other) const;

    virtual tree_node_type diff(const tree_node_type& other) const;
};

class NodeArray : public std::enable_shared_from_this<NodeArray>
{
    std::vector<tree_node_type> m_container_;

public:
    NodeArray() = default;

    virtual ~NodeArray() = default;

    NodeArray(const NodeArray& other) : m_container_(other.m_container_) {}

    NodeArray(NodeArray&& other) : m_container_(std::move(other.m_container_)) {}

    void swap(NodeArray& other) { m_container_.swap(other.m_container_); }

    NodeArray& operator=(const NodeArray& other)
    {
        NodeArray(other).swap(*this);
        return *this;
    }

    //-------------------------------------------------------------------------------
    static std::shared_ptr<NodeArray> create(const std::string& backend = "");

    virtual std::unique_ptr<NodeArray> copy() const { return std::unique_ptr<NodeArray>(new NodeArray(*this)); };

    virtual void clear();

    virtual size_t size() const;

    virtual Cursor<tree_node_type> children();

    virtual Cursor<const tree_node_type> children() const;

    virtual void for_each(std::function<void(int, tree_node_type&)> const&);

    virtual void for_each(std::function<void(int, const tree_node_type&)> const&) const;

    virtual tree_node_type slice(int start, int stop, int step);

    virtual tree_node_type slice(int start, int stop, int step) const;

    virtual void resize(std::size_t num);

    virtual tree_node_type& insert(int idx, tree_node_type);

    virtual tree_node_type& at(int idx);

    virtual const tree_node_type& at(int idx) const;

    virtual tree_node_type& push_back(tree_node_type v = {});

    virtual tree_node_type pop_back();
};

std::ostream& operator<<(std::ostream& os, tree_node_type const& node);

namespace literals
{
using namespace std::complex_literals;
using namespace std::string_literals;
} // namespace literals
} // namespace sp::db

#endif //SP_NODE_H_