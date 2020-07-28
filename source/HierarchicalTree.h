
#ifndef SP_HierarchicalTree_h_
#define SP_HierarchicalTree_h_
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

template <typename TNode>
class HierarchicalTreeObject
{
public:
    typedef TNode node_type;
    typedef HierarchicalTreeObject<node_type> this_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;

    HierarchicalTreeObject() = default;
    HierarchicalTreeObject(this_type&&) = default;
    HierarchicalTreeObject(const this_type&) = default;
    ~HierarchicalTreeObject() = default;

    void swap(this_type& other) { std::swap(m_data_, other.m_data_); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }
    
    size_t size() const { return m_data_.size(); }

    void clear() { m_data_.clear(); }

    int count(const std::string& key) const { return m_data_.count(key); }

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

template <typename TNode>
class HierarchicalTreeArray
{
public:
    typedef TNode node_type;
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

    void resize(size_t n, node_type* parent = nullptr)
    {
        auto num = size();

        return m_data_.resize(n, node_type(parent));
    }

    void clear() { m_data_.clear(); }

    template <typename... Args>
    cursor push_back(Args&&... args)
    {
        m_data_.emplace_back(std::forward<Args>(args)...);
        return cursor(m_data_.rbegin());
    }

    void pop_back() { m_data_.pop_back(); }

    typename cursor::reference at(int idx) { return m_data_.at(idx); }

    typename const_cursor::reference at(int idx) const { return m_data_.at(idx); }

    auto& data() { return m_data_; }

    const auto& data() const { return m_data_; }

private:
    std::vector<node_type> m_data_;
};

/**
 * Hierarchical Tree Struct
*/
template <typename TNode, typename... TypeList>
class HierarchicalTree
{

public:
    static const std::size_t NULL_TAG = 0;

    static const std::size_t OBJECT_TAG = 1;

    static const std::size_t ARRAY_TAG = 2;

    typedef HierarchicalTree<TNode, TypeList...> this_type;

    typedef TNode node_type;

    typedef Cursor<node_type> cursor;

    typedef Cursor<const node_type> const_cursor;

    typedef std::variant<
        std::nullptr_t,
        HierarchicalTreeObject<node_type>,
        HierarchicalTreeArray<node_type>,
        TypeList...>
        data_type;

    friend class HierarchicalTreeArray<node_type>;
    friend class HierarchicalTreeObject<node_type>;

    HierarchicalTree(this_type* p = nullptr, const std::string& name = "") : m_name_(name), m_parent_(p), m_data_(nullptr) {}

    template <int TAG, typename... Args>
    HierarchicalTree(this_type* p, const std::string& name, std::integral_constant<int, TAG>, Args&&... args)
        : m_name_(name), m_parent_(p), m_data_()
    {
        m_data_.template emplace<TAG>(std::forward<Args>(args)...);
    }

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

            m_data_.template emplace<OBJECT_TAG>();
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

    cursor insert(const std::string& key) { return as_object().insert(key, reinterpret_cast<node_type*>(this), key); }

    typename const_cursor::reference at(const std::string& key) const { return *as_object().find(key); }

    typename cursor::reference operator[](const std::string& key) { return *insert(key); }

    typename const_cursor::reference operator[](const std::string& key) const { return at(key); }

    void remove(const std::string& key)
    {
        if (m_data_.index() == OBJECT_TAG)
        {
            as_object().remove(key);
        }
    }

    int count(const std::string& key) const
    {
        return m_data_.index() == OBJECT_TAG && std::get<OBJECT_TAG>(m_data_).count(key);
    }

    template <typename... Args>
    cursor insert(Args&&... args) { return as_object().insert(std::forward<Args>(args)...); }

    template <typename... Args>
    const_cursor find(Args&&... args) const { return as_object().find(std::forward<Args>(args)...); }

    template <typename... Args>
    void remove(Args&&... args) { as_object().remove(std::forward<Args>(args)...); }

    //------------------------------------------------------------------------------
    // as array
    bool is_array() const { return m_data_.index() == ARRAY_TAG; }

    auto& as_array()
    {
        if (m_data_.index() == NULL_TAG)
        {
            static_assert(std::is_base_of_v<this_type, node_type>);

            m_data_.template emplace<ARRAY_TAG>();
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

    void resize(size_t s) { as_array().resize(s, reinterpret_cast<node_type*>(this)); }

    cursor push_back() { return as_array().emplace_back(dynamic_cast<node_type*>(this)); }

    void pop_back() { as_array().pop_back(); }

    typename cursor::reference operator[](int idx) { return as_array().at(idx); }

    typename const_cursor::reference operator[](int idx) const { return as_array()[idx]; }

    //------------------------------------------------------------------------------------------

    data_type& data() { return m_data_; }

    const data_type& data() const { return m_data_; }

    typename cursor::reference operator[](const Path& path) { return as_object().fetch(path); }

    typename const_cursor::reference operator[](const Path& path) const { return as_object().fetch(path); }

    cursor select(const Path& path) { return as_object().select(*this); }

    const_cursor select(const Path& path) const { return as_object().select(*this); }

private:
    this_type* m_parent_;
    std::string m_name_;
    data_type m_data_;
};

template <typename TNode>
using HierarchicalTreePreDefined = HierarchicalTree<
    TNode,
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

enum HierarchicalTreePreDefinedDataType
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

class HierarchicalNode
    : public HierarchicalTreePreDefined<HierarchicalNode>
{
public:
    typedef HierarchicalTreePreDefined<HierarchicalNode> base_type;
    typedef HierarchicalNode this_type;
    typedef HierarchicalTreePreDefinedDataType DataType;

    template <typename... Args>
    HierarchicalNode(Args&&... args) : base_type(std::forward<Args>(args)...) {}
    // using base_type::HierarchicalTree;
    // HNode(HNode&&) = delete;
    // HNode(const HNode&) = delete;
    ~HierarchicalNode() = default;
    template <typename V>
    this_type& operator=(const V& v)
    {
        base_type::operator=(v);
        return *this;
    }
};

} // namespace sp

#endif // SP_HierarchicalTree_h_