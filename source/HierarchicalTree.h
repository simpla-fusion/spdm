
#ifndef SP_HIERACHICAL_DATA_H_
#define SP_HIERACHICAL_DATA_H_
#include "Cursor.h"
#include "utility/Logger.h"
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

/**
 * Hierarchical Tree Struct
*/
template <typename TNode,
          template <typename> class ObjectHolder,
          template <typename> class ArrayHolder,
          typename... TypeList>
class HierarchicalTree
{

public:
    static const std::size_t NULL_TAG = 0;

    static const std::size_t OBJECT_TAG = 1;

    static const std::size_t ARRAY_TAG = 2;

    typedef HierarchicalTree<TNode, ObjectHolder, ArrayHolder, TypeList...> this_type;

    typedef TNode node_type;

    typedef Cursor<node_type> cursor;

    typedef Cursor<const node_type> const_cursor;

    typedef std::variant<
        std::nullptr_t,
        ObjectHolder<node_type>,
        ArrayHolder<node_type>,
        TypeList...>
        data_type;

    HierarchicalTree(this_type* p = nullptr, const std::string& name = "") : m_name_(name), m_parent_(p), m_data_(nullptr) {}

    HierarchicalTree(const this_type& other) : m_parent_(nullptr), m_name_(""), m_data_(other.m_data_){};

    HierarchicalTree(this_type&& other) : m_parent_(nullptr), m_name_(""), m_data_(std::move(other.m_data_)){};

    ~HierarchicalTree() = default;

    void swap(this_type& other) { m_data_.swap(other.m_data_); }

    this_type& operator=(this_type const& other)
    {
        data_type(other.m_data_).swap(m_data_);
        return *this;
    }

    const auto* parent() const { return m_parent_; }

    std::string path() const { return m_parent_ == nullptr ? m_name_ : m_parent_->path() + "/" + m_name_; }

    std::string name() const { return m_name_; }

    auto type() const { return m_data_.index(); }

    bool is_root() const { return m_parent_ == nullptr; }

    bool is_leaf() const { return m_data_.index() != OBJECT_TAG && m_data_.index() != ARRAY_TAG; }

    bool empty() const { return m_data_.index() == NULL_TAG; }

    //---------------------------------------------------------------------------------
    // as leaf

    bool is_element() const { return m_data_.index() > ARRAY_TAG; }

    template <typename V>
    bool operator==(const V& v) const { return m_data_ == data_type(v); }

    template <typename V>
    this_type& operator=(const V& v)
    {
        set_value<V>(v);
        return *this;
    }

    template <typename V, typename... Args>
    void set_value(Args&&... args) { m_data_.template emplace<V>(std::forward<Args>(args)...); }

    template <std::size_t V, typename... Args>
    void set_value(Args&&... args) { m_data_.template emplace<V>(std::forward<Args>(args)...); }

    template <typename V>
    decltype(auto) get_value() { return std::get<V>(m_data_); }

    template <typename V>
    decltype(auto) get_value() const { return std::get<V>(m_data_); }

    template <std::size_t TAG>
    decltype(auto) get_value() { return std::get<TAG>(m_data_); }

    template <std::size_t TAG>
    decltype(auto) get_value() const { return std::get<TAG>(m_data_); }

    //-------------------------------------------------------------------------------
    // as tree node

    void clear()
    {
        if (m_data_.index() == ARRAY_TAG)
        {
            std::get<ARRAY_TAG>(m_data_).clear();
        }
        else if (m_data_.index() == OBJECT_TAG)
        {
            std::get<OBJECT_TAG>(m_data_).clear();
        }
    }

    size_t size() const
    {
        if (m_data_.index() == ARRAY_TAG)
        {
            return std::get<ARRAY_TAG>(m_data_).size();
        }
        else if (m_data_.index() == OBJECT_TAG)
        {
            return std::get<OBJECT_TAG>(m_data_).size();
        }
        else
        {
            return 0;
        }
    }

    //------------------------------------------------------------------------------
    // as object

    bool is_object() const { return m_data_.index() == OBJECT_TAG; }

    auto& as_object()
    {
        if (m_data_.index() == NULL_TAG)
        {
            static_assert(std::is_base_of_v<this_type, node_type>);

            m_data_.template emplace<OBJECT_TAG>(reinterpret_cast<node_type*>(this));
        }
        if (m_data_.index() != OBJECT_TAG)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }

        return std::get<OBJECT_TAG>(m_data_);
    }

    const auto& as_object() const
    {
        if (m_data_.index() != OBJECT_TAG)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }

        return std::get<OBJECT_TAG>(m_data_);
    }

    cursor insert(const std::string& key) { return as_object().insert(key); }

    typename cursor::reference operator[](const std::string& key) { return *insert(key); }

    typename const_cursor::reference at(const std::string& key) const { return as_object().find(key); }

    typename const_cursor::reference operator[](const std::string& key) const { return at(key); }

    void erase(const std::string& key)
    {
        if (m_data_.index() == OBJECT_TAG)
        {
            as_object().erase(key);
        }
    }

    bool has_a(const std::string& key) const
    {
        return m_data_.index() == OBJECT_TAG && std::get<OBJECT_TAG>(m_data_).has_a(key);
    }

    template <typename... Args>
    cursor insert(Args&&... args) { return as_object().insert(std::forward<Args>(args)...); }

    template <typename... Args>
    const_cursor find(Args&&... args) const { return as_object().find(std::forward<Args>(args)...); }

    template <typename... Args>
    void erase(Args&&... args) { as_object().erase(std::forward<Args>(args)...); }

    //------------------------------------------------------------------------------
    // as array
    bool is_array() const { return m_data_.index() == ARRAY_TAG; }

    auto& as_array()
    {
        if (m_data_.index() == NULL_TAG)
        {
            static_assert(std::is_base_of_v<this_type, node_type>);

            m_data_.template emplace<ARRAY_TAG>(reinterpret_cast<node_type*>(this));
        }
        if (m_data_.index() != ARRAY_TAG)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }
        return std::get<ARRAY_TAG>(m_data_);
    }

    const auto& as_array() const
    {
        if (m_data_.index() != ARRAY_TAG)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }
        return std::get<ARRAY_TAG>(m_data_);
    }

    void resize(size_t s) { as_array().resize(s); }

    cursor push_back() { return as_array().emplace_back(m_parent_, m_name_); }

    void pop_back() { as_array().pop_back(); }

    typename cursor::reference operator[](int idx) { return as_array().at(idx); }

    typename const_cursor::reference operator[](int idx) const { return as_array()[idx]; }

    cursor get_r(const std::string& path) { return as_object().insert(path); }

    const_cursor get_r(const std::string& path) const { return as_object().find(path); }

    data_type& data() { return m_data_; }

    const data_type& data() const { return m_data_; }

private:
    this_type* m_parent_;
    std::string m_name_;
    data_type m_data_;
};

template <typename TNode>
class HierarchicalTreeObject
{
public:
    typedef TNode node_type;
    typedef HierarchicalTreeObject<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeObject(node_type* self){};

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

    bool has_a(const std::string& key) const { return m_data_.count(key) > 0; }

    typename const_cursor::reference at(const std::string& key) const { return m_data_.at(key); };

    cursor insert(const std::string& path) { return cursor(m_data_.try_emplace(path, m_self_, path).first); }

    template <typename... Args>
    void erase(Args&&... args) { m_data_.erase(std::forward<Args>(args)...); }

    template <typename... Args>
    cursor find(Args&&... args) { return cursor(m_data_.find(std::forward<Args>(args)...)); }

    template <typename... Args>
    const_cursor find(Args&&... args) const { return cursor(m_data_.find(std::forward<Args>(args)...)); }

    auto& data() { return m_data_; }

    const auto& data() const { return m_data_; }

private:
    std::map<std::string, node_type> m_data_;
    node_type* m_self_;
};

template <typename TNode>
class HierarchicalTreeArray
{
public:
    typedef TNode node_type;
    typedef HierarchicalTreeArray<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeArray(node_type* self){};
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
        m_data_.emplace_back(m_self_, std::forward<Args>(args)...);
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
    node_type* m_self_;
};

template <typename TNode,
          template <typename> class ObjectHolder,
          template <typename> class ArrayHolder>
using HierarchicalTreePreDefined = HierarchicalTree<
    TNode,
    ObjectHolder,                                                //Object
    ArrayHolder,                                                 //Array
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
    >;

template <template <typename> class ObjectHolder = HierarchicalTreeObject,
          template <typename> class ArrayHolder = HierarchicalTreeArray>
class HierarchicalNode
    : public HierarchicalTreePreDefined<HierarchicalNode<ObjectHolder, ArrayHolder>, ObjectHolder, ArrayHolder>
{
public:
    typedef HierarchicalTreePreDefined<HierarchicalNode<ObjectHolder, ArrayHolder>, ObjectHolder, ArrayHolder> base_type;

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
    HierarchicalNode(Args&&... args) : base_type(std::forward<Args>(args)...) {}
    // using base_type::HierarchicalTree;
    // HierarchicalNode(HierarchicalNode&&) = delete;
    // HierarchicalNode(const HierarchicalNode&) = delete;
    ~HierarchicalNode() = default;
    template <typename V>
    HierarchicalNode& operator=(const V& v)
    {
        base_type::operator=(v);
        return *this;
    }
};

} // namespace sp

#endif //SP_HIERACHICAL_DATA_H_