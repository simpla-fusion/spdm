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
class Entry;
class Node;
template <>
struct cursor_traits<Node>
{
    typedef Node value_type;
    typedef Node& reference;
    typedef Node* pointer;
    // typedef Node reference;
    // typedef std::shared_ptr<Node> pointer;
    typedef ptrdiff_t difference_type;
};

template <>
class HierarchicalTreeObject<Node>
{
public:
    typedef Node node_type;
    typedef HierarchicalTreeObject<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeObject(const std::string&){};
    HierarchicalTreeObject() = default;
    HierarchicalTreeObject(this_type&&) = default;
    HierarchicalTreeObject(const this_type&) = default;
    ~HierarchicalTreeObject() = default;

    void swap(this_type& other) { m_data_.swap(other.m_data_); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }
    size_t size() const { return m_data_.size(); }

    void clear() { m_data_.clear(); }

    int count(const std::string& key) const { return m_data_.count(key); }

    typename const_cursor::reference at(const std::string& key) const { return m_data_.at(key); };

    template <typename... Args>
    cursor insert(const std::string& path, Args&&... args) { return cursor(m_data_.try_emplace(path, std::forward<Args>(args)...).first); }

    template <typename... Args>
    void remove(Args&&... args) { m_data_.erase(std::forward<Args>(args)...); }

    template <typename... Args>
    cursor find(Args&&... args) { return cursor(m_data_.find(std::forward<Args>(args)...)); }

    template <typename... Args>
    const_cursor find(Args&&... args) const { return cursor(m_data_.find(std::forward<Args>(args)...)); }

    auto& data() { return m_data_; }

    const auto& data() const { return m_data_; }

private:
    std::map<std::string, node_type> m_data_;
};

template <>
class HierarchicalTreeArray<Node>
{
public:
    typedef Node node_type;
    typedef HierarchicalTreeArray<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeArray() = default;
    HierarchicalTreeArray(this_type&&) = default;
    HierarchicalTreeArray(const this_type&) = default;
    ~HierarchicalTreeArray() = default;

    void swap(this_type& other) { m_data_.swap(other.m_data_); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    size_t size() const { return m_data_.size(); }

    void resize(size_t s) { m_data_.resize(s); }

    void clear() { m_data_.clear(); }

    template <typename... Args>
    cursor emplace_back(Args&&... args)
    {
        m_data_.emplace_back(std::forward<Args>(args)...);
        return cursor(m_data_.rbegin());
    }

    cursor push_back() { return emplace_back(); }

    void pop_back() { m_data_.pop_back(); }

    typename cursor::reference at(int idx) { return m_data_.at(idx); }

    typename const_cursor::reference at(int idx) const { return m_data_.at(idx); }

    auto& data() { return m_data_; }

    const auto& data() const { return m_data_; }

private:
    std::vector<node_type> m_data_;
};

class Node : public HierarchicalTreePreDefined<Node, HierarchicalTreeObject, HierarchicalTreeArray>
{
public:
    typedef HierarchicalTreePreDefined<Node, HierarchicalTreeObject, HierarchicalTreeArray> base_type;

    typedef Node this_type;

    typedef HierarchicalTreePreDefinedDataType DataType;

    Node(Node* parent = nullptr, const std::string& name = "") : base_type(parent, name) {}

    Node(const std::string& backend)
        : base_type(nullptr, "", std::integral_constant<int, base_type::OBJECT_TAG>(), backend) {}
    //base_type(nullptr, "", std::integral_constant<int,DataType::Object>(), name) {}

    // template <typename... Args>
    // Node(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    ~Node() = default;

    this_type& operator=(Node const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    // attributes

    bool has_attribute(const std::string& name) const { return count("@" + name) > 0; }

    void remove_attribute(const std::string& name) { remove("@" + name); }

    template <typename V>
    auto get_attribute(const std::string& name) { return this->operator[]("@" + name).get_value<V>(); };

    template <typename V, typename U>
    void set_attribute(const std::string& name, const U& value) { this->operator[]("@" + name).set_value<V>(value); }

    // private:
    //     this_type* m_parent_;
    //     std::string m_name_;
    //     data_type& fetch()
    //     {
    //         if (m_data_.index() == NULL_TAG && m_parent_ != nullptr && m_name_ != "")
    //         {
    //             m_parent_->fetch(m_name_).swap(m_data_);
    //         }

    //         return m_data_;
    //     }

    //     const data_type& fetch() const { return const_cast<this_type*>(this)->fetch(); }

    //     data_type fetch(const std::string& path) { return get_r(path).m_data_; }

    //     const data_type& fetch(const std::string& path) const { return get_r(path).m_data_; }
};

std::ostream& operator<<(std::ostream& os, Node const& Node);

} // namespace sp

#endif // SP_NODE_H_
