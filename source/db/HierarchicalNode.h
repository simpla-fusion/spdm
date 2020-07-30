
#ifndef SPDB_HierarchicalNode_h_
#define SPDB_HierarchicalNode_h_
#include "./utility/Logger.h"
#include "./utility/Path.h"
#include "Cursor.h"
#include "HierarchicalTree.h"
#include <any>
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
namespace sp::db
{
class HierarchicalNode;

using HierarchicalNodeBase = HierarchicalTree<
    HierarchicalNode,
    std::map<std::string, HierarchicalNode>,
    std::vector<HierarchicalNode>,
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
    std::any>;

class HierarchicalNode : public HierarchicalNodeBase
{
public:
    typedef HierarchicalNode this_type;
    typedef HierarchicalNodeBase base_type;
    template <typename... Args>
    HierarchicalNode(Args&&... args) : HierarchicalTree(std::forward<Args>(args)...) {}
    ~HierarchicalNode() = default;

    template <typename V>
    this_type& operator=(const V& v)
    {
        base_type::operator=(v);
        return *this;
    }
};
using HTNodeObject = HTContainerProxyObject<HierarchicalNode, std::map<std::string, HierarchicalNode>>;
template <>
inline HTNodeObject::HTContainerProxyObject(node_type* self, container* d) : m_self_(self), m_container_(d != nullptr ? d : new container) {}
template <>
inline HTNodeObject::HTContainerProxyObject(this_type&& other) : this_type(other.m_self_, other.m_container_.release()) {}
template <>
inline HTNodeObject::HTContainerProxyObject(const this_type& other) : this_type(nullptr, new container(*other.m_container_)) {}

template <>
inline HTNodeObject::~HTContainerProxyObject() {}

template <>
inline size_t HTNodeObject::size() const { return m_container_->size(); }

template <>
inline void HTNodeObject::clear() { m_container_->clear(); }

template <>
inline int HTNodeObject::count(const std::string& key) const { return m_container_->count(key); }

template <>
inline HTNodeObject::cursor
HTNodeObject::insert(const std::string& path) { return cursor(m_container_->try_emplace(path, m_self_, path).first); }

template <>
inline HTNodeObject::cursor
HTNodeObject::insert(const Path& path) { return insert(path.str()); }

template <>
inline void HTNodeObject::erase(const std::string& path) { m_container_->erase(path); }

template <>
inline void HTNodeObject::erase(const Path& path) { erase(path.str()); }

template <>
inline HTNodeObject::cursor
HTNodeObject::find(const std::string& path) { return cursor(m_container_->find(path)); }

template <>
inline HTNodeObject::cursor
HTNodeObject::find(const Path& path) { return (find(path.str())); }

template <>
inline HTNodeObject::const_cursor
HTNodeObject::find(const std::string& path) const { return const_cursor(m_container_->find(path)); }

template <>
inline HTNodeObject::const_cursor
HTNodeObject::find(const Path& path) const { return (find(path.str())); }

//-----------------------------------------------------------------------------------
// Array
using HTNodeArray = HTContainerProxyArray<HierarchicalNode, std::vector<HierarchicalNode>>;

template <>
inline HTNodeArray::HTContainerProxyArray(node_type* self, container* d) : m_self_(self), m_container_(d != nullptr ? d : new container) {}
template <>
inline HTNodeArray::HTContainerProxyArray(this_type&& other) : this_type(nullptr, other.m_container_.release()) {}
template <>
inline HTNodeArray::HTContainerProxyArray(const this_type& other) : this_type(nullptr, new container(*other.m_container_)) {}
template <>
inline HTNodeArray::~HTContainerProxyArray() {}

template <>
inline size_t HTNodeArray::size() const { return m_container_->size(); }

template <>
inline void HTNodeArray::resize(std::size_t num) { m_container_->resize(num, node_type(m_self_)); }

template <>
inline void HTNodeArray::clear() { m_container_->clear(); }

template <>
inline HTNodeArray::cursor
HTNodeArray::push_back()
{
    m_container_->emplace_back(m_self_);
    return cursor(m_container_->rbegin());
}

template <>
inline void HTNodeArray::pop_back() { m_container_->pop_back(); }

template <>
inline typename Cursor<HierarchicalNode>::reference
HTNodeArray::at(int idx) { return m_container_->at(idx); }

template <>
inline typename Cursor<const HierarchicalNode>::reference
HTNodeArray::at(int idx) const { return m_container_->at(idx); }
} // namespace sp::db

#endif // SPDB_HierarchicalNode_h_