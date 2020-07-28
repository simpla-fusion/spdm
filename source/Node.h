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

template <typename TNode>
class HierarchicalTreeObject
{
public:
    typedef Node node_type;
    typedef HierarchicalTreeObject<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeObject();
    HierarchicalTreeObject(this_type&&);
    HierarchicalTreeObject(const this_type&);
    ~HierarchicalTreeObject() = default;
    HierarchicalTreeObject(const std::string&);

    void swap(this_type& other);

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    size_t size() const;

    void clear();

    int count(const std::string& key) const { return m_data_.count(key); }

    int count(const Path& key) const { return m_data_.count(key); }

    cursor insert(const std::string& path, node_type* self);

    cursor insert(const Path& path, node_type* self);

    void remove(const std::string& path);

    void remove(const Path& path);

    cursor find(const std::string& path);

    cursor find(const Path& path);

    const_cursor find(const Path& path) const;

    const_cursor find(const std::string& path) const;

    cursor find(const Path& path);

    auto& data() { return m_data_; }

    const auto& data() const { return m_data_; }

private:
    std::shared_ptr<Entry> m_data_;
};

template <typename TNode>
class HierarchicalTreeArray
{
public:
    typedef Node node_type;
    typedef HierarchicalTreeArray<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeArray();
    HierarchicalTreeArray(this_type&&);
    HierarchicalTreeArray(const this_type&);
    ~HierarchicalTreeArray() = default;

    void swap(this_type& other)

        this_type&
        operator=(this_type const& other);

    size_t size() const;

    void resize(size_t s);

    void clear();

    cursor push_back(node_type* self);

    void pop_back();

    typename cursor::reference at(int idx);

    typename const_cursor::reference at(int idx) const;

    auto& data() { return m_data_; }

    const auto& data() const { return m_data_; }

private:
    std::shared_ptr<Entry> m_data_;
};

class Node : public HierarchicalTreePreDefined<Node>
{
public:
    typedef HierarchicalTreePreDefined<Node> base_type;

    typedef Node this_type;

    typedef HierarchicalTreePreDefinedDataType DataType;

    Node(Node* parent = nullptr, const std::string& name = "");

    Node(const std::string& backend);

    ~Node() = default;

    this_type& operator=(Node const& other);

    // attributes

    bool has_attribute(const std::string& name) const { return count("@" + name) > 0; }

    void remove_attribute(const std::string& name) { remove("@" + name); }

    template <typename V>
    auto get_attribute(const std::string& name) { return find("@" + name)->get_value<V>(); };

    template <typename V, typename U>
    void set_attribute(const std::string& name, const U& value) { this->insert<V>("@" + name, value); }

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
