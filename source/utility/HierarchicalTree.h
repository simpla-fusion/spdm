
#ifndef SP_HierarchicalTree_h_
#define SP_HierarchicalTree_h_
#include "Cursor.h"
#include "Logger.h"
#include "Path.h"
#include "TypeTraits.h"
#include "fancy_print.h"
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
struct node_traits
{
    typedef TNode node_type;
    typedef Cursor<node_type> cursor;
    typedef Cursor<const node_type> const_cursor;
    typedef node_type& reference;
    typedef node_type* pointer;
    typedef std::map<std::string, node_type> object_container;
    typedef std::vector<node_type> array_container;
};

template <typename TNode>
class HierarchicalTreeObjectContainer;
template <typename TNode>
class HierarchicalTreeArrayContainer;

/**
 * Hierarchical Tree Struct
*/
template <typename TNode, typename... TypeList>
class HierarchicalTree
{

public:
    typedef TNode node_type;

    typedef HierarchicalTree<node_type, TypeList...> this_type;

    typedef HierarchicalTree<node_type, TypeList...> tree_type;

    typedef typename node_traits<node_type>::cursor cursor;

    typedef typename node_traits<node_type>::const_cursor const_cursor;

    typedef std::variant<
        std::nullptr_t,
        HierarchicalTreeObjectContainer<node_type>,
        HierarchicalTreeArrayContainer<node_type>,
        TypeList...>
        type_union;

    struct _head
    {
        enum
        {
            Empty = 0,
            Object = 1,
            Array = 2,
            _LAST_PLACE_HOLDER
        };
    };

    typedef typename traits::type_tag_traits<_head, TypeList...>::tags type_tag;

    friend class Array;
    friend class Object;

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
        type_union(other.m_data_).swap(m_data_);
        return *this;
    }

    auto parent() const { return m_parent_; }

    std::string path() const { return m_parent_ == nullptr ? m_name_ : m_parent_->path() + "/" + m_name_; }

    std::string name() const { return m_name_; }

    auto type() const { return m_data_.index(); }

    bool is_root() const { return m_parent_ == nullptr; }

    bool is_leaf() const { return m_data_.index() != type_tag::Object && m_data_.index() != type_tag::Array; }

    bool empty() const { return m_data_.index() == type_tag::Empty; }

    //---------------------------------------------------------------------------------
    // as leaf

    bool is_element() const { return m_data_.index() > type_tag::Array; }

    template <typename V>
    bool operator==(const V& v) const { return m_data_ == type_union(v); }

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
        if (m_data_.index() == type_tag::Array)
        {
            std::get<type_tag::Array>(m_data_).clear();
        }
        else if (m_data_.index() == type_tag::Object)
        {
            std::get<type_tag::Object>(m_data_).clear();
        }
    }

    size_t size() const
    {
        if (m_data_.index() == type_tag::Array)
        {
            return std::get<type_tag::Array>(m_data_).size();
        }
        else if (m_data_.index() == type_tag::Object)
        {
            return std::get<type_tag::Object>(m_data_).size();
        }
        else
        {
            return 0;
        }
    }

    //------------------------------------------------------------------------------
    // as object

    bool is_object() const { return m_data_.index() == type_tag::Object; }

    decltype(auto) as_object()
    {
        if (m_data_.index() == type_tag::Empty)
        {
            static_assert(std::is_base_of_v<this_type, node_type>);

            m_data_.template emplace<type_tag::Object>();
        }
        if (m_data_.index() != type_tag::Object)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }

        return std::get<type_tag::Object>(m_data_);
    }

    decltype(auto) as_object() const
    {
        if (m_data_.index() != type_tag::Object)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }

        return std::get<type_tag::Object>(m_data_);
    }

    auto insert(const std::string& key) { return as_object().insert(key, reinterpret_cast<node_type*>(this)); }

    decltype(auto) at(const std::string& key) const { return *as_object().find(key); }

    decltype(auto) operator[](const std::string& key) { return *insert(key); }

    decltype(auto) operator[](const std::string& key) const { return at(key); }

    void erase(const std::string& key)
    {
        if (m_data_.index() == type_tag::Object)
        {
            as_object().erase(key);
        }
    }

    int count(const std::string& key) const
    {
        return m_data_.index() == type_tag::Object && std::get<type_tag::Object>(m_data_).count(key);
    }

    template <typename... Args>
    auto insert(Args&&... args) { return as_object().insert(std::forward<Args>(args)...); }

    template <typename... Args>
    auto find(Args&&... args) const { return as_object().find(std::forward<Args>(args)...); }

    template <typename... Args>
    void erase(Args&&... args) { as_object().erase(std::forward<Args>(args)...); }

    //------------------------------------------------------------------------------
    // as array
    bool is_array() const { return m_data_.index() == type_tag::Array; }

    decltype(auto) as_array()
    {
        if (m_data_.index() == type_tag::Empty)
        {
            static_assert(std::is_base_of_v<this_type, node_type>);

            m_data_.template emplace<type_tag::Array>();
        }
        if (m_data_.index() != type_tag::Array)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }
        return std::get<type_tag::Array>(m_data_);
    }

    decltype(auto) as_array() const
    {
        if (m_data_.index() != type_tag::Array)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING);
        }
        return std::get<type_tag::Array>(m_data_);
    }

    void resize(size_t s) { as_array().resize(s, reinterpret_cast<node_type*>(this)); }

    auto push_back() { return as_array().emplace_back(dynamic_cast<node_type*>(this)); }

    void pop_back() { as_array().pop_back(); }

    decltype(auto) operator[](int idx) { return as_array().at(idx); }

    decltype(auto) operator[](int idx) const { return as_array()[idx]; }

    //------------------------------------------------------------------------------------------

    type_union& data() { return m_data_; }

    const type_union& data() const { return m_data_; }

    decltype(auto) operator[](const Path& path) { return *as_object().find(path); }

    decltype(auto) operator[](const Path& path) const { return *as_object().find(path); }

    auto select(const Path& path) { return as_object().select(path); }

    auto select(const Path& path) const { return as_object().select(path); }

private:
    this_type* m_parent_;
    std::string m_name_;
    type_union m_data_;
};

template <typename TNode>
class HierarchicalTreeObjectContainer
{
public:
    typedef TNode node_type;
    typedef HierarchicalTreeObjectContainer<node_type> this_type;
    typedef node_traits<node_type> traits_type;
    typedef typename node_traits<node_type>::cursor cursor;
    typedef typename node_traits<node_type>::const_cursor const_cursor;
    typedef typename node_traits<node_type>::object_container container;

    HierarchicalTreeObjectContainer(container* container);
    HierarchicalTreeObjectContainer();
    HierarchicalTreeObjectContainer(this_type&& other);
    HierarchicalTreeObjectContainer(const this_type& other);
    ~HierarchicalTreeObjectContainer();

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type& other) { std::swap(m_container_, other.m_container_); }

    size_t size() const;

    void clear();

    int count(const std::string& key) const;

    cursor insert(const std::string& path, node_type* self);

    cursor insert(const Path& path, node_type* self);

    void erase(const std::string& path);

    void erase(const Path& path);

    cursor find(const std::string& path);

    cursor find(const Path& path);

    const_cursor find(const std::string& path) const;

    const_cursor find(const Path& path) const;

private:
    std::unique_ptr<container> m_container_;
};

template <typename TNode>
class HierarchicalTreeArrayContainer
{
public:
    typedef TNode node_type;
    typedef HierarchicalTreeArrayContainer<node_type> this_type;
    typedef node_traits<node_type> traits_type;
    typedef typename node_traits<node_type>::cursor cursor;
    typedef typename node_traits<node_type>::const_cursor const_cursor;
    typedef typename node_traits<node_type>::array_container container;

    HierarchicalTreeArrayContainer(container* container);
    HierarchicalTreeArrayContainer();
    HierarchicalTreeArrayContainer(this_type&& other);
    HierarchicalTreeArrayContainer(const this_type& other);
    ~HierarchicalTreeArrayContainer();

    void swap(this_type& other) { std::swap(m_container_, other.m_container_); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    size_t size() const;

    void resize(std::size_t num, node_type* self);

    void clear();

    cursor push_back(node_type* self);

    void pop_back();

    typename cursor::reference at(int idx);

    typename const_cursor::reference at(int idx) const;

private:
    std::unique_ptr<container> m_container_;
};

template <typename TNode, typename... TypeList>
std::ostream& fancy_print(std::ostream& os, const typename HierarchicalTree<TNode, TypeList...>::Object& tree_object, int indent, int tab)
{
    return fancy_print(os, tree_object.data(), indent, tab);
}
template <typename TNode, typename... TypeList>
std::ostream& fancy_print(std::ostream& os, const typename HierarchicalTree<TNode, TypeList...>::Array& tree_array, int indent, int tab)
{
    return fancy_print(os, tree_array.data(), indent, tab);
}

template <typename TNode, typename... TypeList>
std::ostream& fancy_print(std::ostream& os, const HierarchicalTree<TNode, TypeList...>& tree, int indent, int tab)
{
    std::visit([&](auto&& v) { fancy_print(os, v, indent, tab); }, tree.data());

    return os;
}

template <typename TNode, typename... TypeList>
std::ostream& operator<<(std::ostream& os, const HierarchicalTree<TNode, TypeList...>& tree)
{
    return fancy_print(os, tree, 0, 4);
}

// enum
// {
//     Block = hierarchical_tree_type_traits<>::type_tag::_LAST_PLACE_HOLDER,
//     String,
//     Bool,
//     Int,
//     Long,
//     Float,
//     Double,
//     Complex,
//     IntVec3,
//     LongVec3,
//     FloatVec3,
//     DoubleVec3,
//     ComplexVec3,
//     UNKNOWN,
//     _LAST_PLACE_HOLDER
// };

// template <>
// struct hierarchical_tree_type_traits<
//     std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
//     std::string,                                                 //String,
//     bool,                                                        //Boolean,
//     int,                                                         //Integer,
//     long,                                                        //Long,
//     float,                                                       //Float,
//     double,                                                      //Double,
//     std::complex<double>,                                        //Complex,
//     std::array<int, 3>,                                          //IntVec3,
//     std::array<long, 3>,                                         //LongVec3,
//     std::array<float, 3>,                                        //FloatVec3,
//     std::array<double, 3>,                                       //DoubleVec3,
//     std::array<std::complex<double>, 3>,                         //ComplexVec3,
//     std::any                                                     //Other
//     >

// template <>
// struct hierarchical_tree_type_traits<
//     std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
//     std::string,                                                 //String,
//     bool,                                                        //Boolean,
//     int,                                                         //Integer,
//     long,                                                        //Long,
//     float,                                                       //Float,
//     double,                                                      //Double,
//     std::complex<double>,                                        //Complex,
//     std::array<int, 3>,                                          //IntVec3,
//     std::array<long, 3>,                                         //LongVec3,
//     std::array<float, 3>,                                        //FloatVec3,
//     std::array<double, 3>,                                       //DoubleVec3,
//     std::array<std::complex<double>, 3>,                         //ComplexVec3,
//     std::any                                                     //Other
//     >
// {
//     typedef std::variant<std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
//                          std::string,                                                 //String,
//                          bool,                                                        //Boolean,
//                          int,                                                         //Integer,
//                          long,                                                        //Long,
//                          float,                                                       //Float,
//                          double,                                                      //Double,
//                          std::complex<double>,                                        //Complex,
//                          std::array<int, 3>,                                          //IntVec3,
//                          std::array<long, 3>,                                         //LongVec3,
//                          std::array<float, 3>,                                        //FloatVec3,
//                          std::array<double, 3>,                                       //DoubleVec3,
//                          std::array<std::complex<double>, 3>,                         //ComplexVec3,
//                          std::any                                                     //Other
//                          >
//         type_union;

//     struct type_tag : public hierarchical_tree_type_traits<>::type_tag
//     {

//         enum
//         {
//             Block = hierarchical_tree_type_traits<>::type_tag::_LAST_PLACE_HOLDER,
//             String,
//             Bool,
//             Int,
//             Long,
//             Float,
//             Double,
//             Complex,
//             IntVec3,
//             LongVec3,
//             FloatVec3,
//             DoubleVec3,
//             ComplexVec3,
//             UNKNOWN,
//             _LAST_PLACE_HOLDER
//         };
//     };

// };

} // namespace sp

#endif // SP_HierarchicalTree_h_