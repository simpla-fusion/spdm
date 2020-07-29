
#ifndef SP_HierarchicalNode_h_
#define SP_HierarchicalNode_h_
#include "HierarchicalTree.h"
#include "utility/Cursor.h"
#include "utility/Logger.h"
#include "utility/Path.h"
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
namespace sp
{

class HierarchicalNode
    : public HierarchicalTree<HierarchicalNode,
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
                              std::any>
{
public:
    typedef HierarchicalNode this_type;

    template <typename... Args>
    HierarchicalNode(Args&&... args) : HierarchicalTree(std::forward<Args>(args)...) {}
    ~HierarchicalNode() = default;

    template <typename V>
    this_type& operator=(const V& v)
    {
        tree_type::operator=(v);
        return *this;
    }
};

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::HierarchicalTreeObjectContainer(node_type* self, container* d) : m_self_(self), m_container_(d != nullptr ? d : new container) {}
template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::HierarchicalTreeObjectContainer(this_type&& other) : this_type(other.m_self_, other.m_container_.release()) {}
template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::HierarchicalTreeObjectContainer(const this_type& other) : this_type(nullptr, new container(*other.m_container_)) {}

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::~HierarchicalTreeObjectContainer() {}

template <>
inline size_t HierarchicalTreeObjectContainer<HierarchicalNode>::size() const { return m_container_->size(); }

template <>
inline void HierarchicalTreeObjectContainer<HierarchicalNode>::clear() { m_container_->clear(); }

template <>
inline int HierarchicalTreeObjectContainer<HierarchicalNode>::count(const std::string& key) const { return m_container_->count(key); }

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::cursor
HierarchicalTreeObjectContainer<HierarchicalNode>::insert(const std::string& path) { return cursor(m_container_->try_emplace(path, m_self_, path).first); }

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::cursor
HierarchicalTreeObjectContainer<HierarchicalNode>::insert(const Path& path) { return insert(path.str()); }

template <>
inline void HierarchicalTreeObjectContainer<HierarchicalNode>::erase(const std::string& path) { m_container_->erase(path); }

template <>
inline void HierarchicalTreeObjectContainer<HierarchicalNode>::erase(const Path& path) { erase(path.str()); }

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::cursor
HierarchicalTreeObjectContainer<HierarchicalNode>::find(const std::string& path) { return cursor(m_container_->find(path)); }

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::cursor
HierarchicalTreeObjectContainer<HierarchicalNode>::find(const Path& path) { return (find(path.str())); }

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::const_cursor
HierarchicalTreeObjectContainer<HierarchicalNode>::find(const std::string& path) const { return const_cursor(m_container_->find(path)); }

template <>
inline HierarchicalTreeObjectContainer<HierarchicalNode>::const_cursor
HierarchicalTreeObjectContainer<HierarchicalNode>::find(const Path& path) const { return (find(path.str())); }

//-----------------------------------------------------------------------------------
// Array

template <>
inline HierarchicalTreeArrayContainer<HierarchicalNode>::HierarchicalTreeArrayContainer(node_type* self, container* d) : m_self_(self), m_container_(d!=nullptr?d:new container) {}
template <>
inline HierarchicalTreeArrayContainer<HierarchicalNode>::HierarchicalTreeArrayContainer(this_type&& other) : this_type(nullptr, other.m_container_.release()) {}
template <>
inline HierarchicalTreeArrayContainer<HierarchicalNode>::HierarchicalTreeArrayContainer(const this_type& other) : this_type(nullptr, new container(*other.m_container_)) {}
template <>
inline HierarchicalTreeArrayContainer<HierarchicalNode>::~HierarchicalTreeArrayContainer() {}

template <>
inline size_t HierarchicalTreeArrayContainer<HierarchicalNode>::size() const { return m_container_->size(); }

template <>
inline void HierarchicalTreeArrayContainer<HierarchicalNode>::resize(std::size_t num) { m_container_->resize(num, node_type(m_self_)); }

template <>
inline void HierarchicalTreeArrayContainer<HierarchicalNode>::clear() { m_container_->clear(); }

template <>
inline HierarchicalTreeArrayContainer<HierarchicalNode>::cursor
HierarchicalTreeArrayContainer<HierarchicalNode>::push_back()
{
    m_container_->emplace_back(m_self_);
    return cursor(m_container_->rbegin());
}

template <>
inline void HierarchicalTreeArrayContainer<HierarchicalNode>::pop_back() { m_container_->pop_back(); }

template <>
inline typename node_traits<HierarchicalNode>::reference
HierarchicalTreeArrayContainer<HierarchicalNode>::at(int idx) { return m_container_->at(idx); }

template <>
inline typename node_traits<const HierarchicalNode>::reference
HierarchicalTreeArrayContainer<HierarchicalNode>::at(int idx) const { return m_container_->at(idx); }
} // namespace sp

#endif // SP_HierarchicalNode_h_