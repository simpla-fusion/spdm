#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
#include "../utility/Path.h"
#include "../utility/TypeTraits.h"
#include "HierarchicalTree.h"
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
namespace sp
{
namespace db
{
class Entry;
class EntryObject;
class EntryArray;
} // namespace db
} // namespace sp
M_REGISITER_TYPE_TAG(Object, sp::db::EntryObject);
M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, sp::db::EntryArray);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::EntryArray>);

namespace sp
{
namespace db
{

using element_types = std::variant<
    // std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
    std::string, //String,
    bool,        //Boolean,
    int,         //Integer,
    long,        //Long,
    float,       //Float,
    double       //,      //Double,
    // std::complex<double>,                                        //Complex,
    // std::array<int, 3>,                                          //IntVec3,
    // std::array<long, 3>,                                         //LongVec3,
    // std::array<float, 3>,                                        //FloatVec3,
    // std::array<double, 3>,                                       //DoubleVec3,
    // std::array<std::complex<double>, 3>,                         //ComplexVec3,
    //std::any
    >;

typedef traits::template_copy_type_args<
    HierarchicalTree,
    traits::concatenate_t<
        std::variant<Entry,
                     EntryObject,
                     EntryArray>,
        element_types>>
    entry_base;

class EntryObject : std::enable_shared_from_this<EntryObject>
{
public:
    EntryObject() = default;
    EntryObject(const EntryObject&) = delete;
    EntryObject(EntryObject&&) = delete;
    virtual ~EntryObject();

    static std::shared_ptr<EntryObject> create(const std::string& request = "");

    static bool add_creator(const std::string& c_id, const std::function<EntryObject*()>&);

    virtual std::shared_ptr<EntryObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    virtual std::size_t count(const std::string& name) = 0;

    virtual Cursor<Entry> insert(const std::string& path) = 0;

    virtual Cursor<Entry> insert(const Path& path) = 0;

    virtual Cursor<Entry> find(const std::string& path) = 0;

    virtual Cursor<Entry> find(const Path& path) = 0;

    virtual Cursor<const Entry> find(const std::string& path) const = 0;

    virtual Cursor<const Entry> find(const Path& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual void erase(const Path& path) = 0;

    virtual Cursor<Entry> first_child() = 0;

    virtual Cursor<const Entry> first_child() const = 0;

    // level 1

    virtual Cursor<Entry> select(const std::string& path) = 0;

    virtual Cursor<Entry> select(const Path& path) = 0;

    virtual Cursor<const Entry> select(const std::string& path) const = 0;

    virtual Cursor<const Entry> select(const Path& path) const = 0;

    template <typename P>
    Entry& operator[](const P& path) { return *insert(path); }

    template <typename P>
    const Entry& operator[](const P& path) const { return *find(path); }
};

class EntryArray : std::enable_shared_from_this<EntryArray>
{
public:
    EntryArray() = default;

    EntryArray(const EntryArray&) = delete;

    EntryArray(EntryArray&&) = delete;

    virtual ~EntryArray();

    static std::shared_ptr<EntryArray> create(const std::string& request = "");

    virtual std::shared_ptr<EntryArray> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void resize(std::size_t num) = 0;

    virtual void clear() = 0;

    virtual Cursor<Entry> push_back() = 0;

    virtual void pop_back() = 0;

    virtual Entry& at(int idx) = 0;

    virtual const Entry& at(int idx) const = 0;

    Entry& operator[](int idx) { return at(idx); }

    const Entry& operator[](int idx) const { return at(idx); }
};

class Entry : public entry_base
{
public:
    typedef entry_base base_type;
    typedef traits::type_tags<entry_base> type_tags;
    typedef Cursor<Entry> cursor;
    typedef Cursor<const Entry> const_cursor;

    Entry() = default;

    Entry(const Entry& other) : base_type(other) {}

    Entry(Entry&& other) : base_type(std::move(other)) { ; }

    ~Entry() = default;

    void swap(Entry& other) { base_type::swap(other); }

    Entry& operator=(const Entry& other)
    {
        Entry(other).swap(*this);
        return *this;
    }

    std::size_t type() const { return base_type::index(); }

    template <typename V>
    Entry& operator=(const V& v)
    {
        as<V>(v);
        return *this;
    }

    void clear() { emplace<std::nullptr_t>(nullptr); }

    template <typename V>
    void as(const V& v) { emplace<V>(v); }

    template <typename V>
    void as(V&& v) { emplace<V>(std::forward<V>(v)); }

    template <typename V>
    V& as() { return std::get<V>(*this); }

    template <typename V>
    const V& as() const { return std::get<V>(*this); }

    EntryObject& as_object();

    const EntryObject& as_object() const;

    EntryArray& as_array();

    const EntryArray& as_array() const;

    Entry& operator[](const std::string& key) { return as_object()[key]; }

    const Entry& operator[](const std::string& key) const { return as_object()[key]; }

    Entry& operator[](int idx) { return as_array()[idx]; }

    const Entry& operator[](int idx) const { return as_array()[idx]; }
};

// std::string to_string(Entry const& s);

// Entry from_string(const std::string& s, int idx = 0);

template <typename TNode>
using entry_wrapper = traits::template_copy_type_args<
    HierarchicalTree,
    traits::concatenate_t<
        std::variant<TNode,
                     EntryObject,
                     EntryArray>,
        element_types>>;

} // namespace db
} // namespace sp

#endif //SP_ENTRY_H_