#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
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
class Entry;
class EntryNode;
class EntryObject;
class EntryArray;
class DataBlock;

typedef std::pair<std::weak_ptr<EntryObject>, Path> EntryReference;

template <typename E>
using EntryItem = std::pair<const std::string, E>;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::EntryArray>);
M_REGISITER_TYPE_TAG(Block, sp::db::DataBlock);
M_REGISITER_TYPE_TAG(Reference, sp::db::EntryReference);
// M_REGISITER_TYPE_TAG_TEMPLATE(Item, sp::db::EntryItem)
namespace sp::traits
{
template <typename _Head, typename E>
struct regisiter_type_tag<_Head, std::pair<const std::string, E>>
{
    struct tags : public _Head
    {
        enum
        {
            Item = _Head::_LAST_PLACE_HOLDER,
            _LAST_PLACE_HOLDER
        };
    };
};
} // namespace sp::traits

namespace sp::db
{

using entry_value_union = std::variant<std::nullptr_t,
                                       std::shared_ptr<EntryObject>,
                                       std::shared_ptr<EntryArray>,
                                       EntryReference,                     //Reference
                                                                           //    std::pair<const std::string, E>,    //Item
                                       DataBlock,                          //Block
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
                                       >;

class EntryObject
{

public:
    EntryObject() = default;

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    virtual ~EntryObject() = default;

    //-------------------------------------------------------------------------------

    virtual std::unique_ptr<EntryObject> copy() const;

    virtual size_t size() const;

    virtual void clear();

    virtual Cursor<Entry> children();

    virtual Cursor<const Entry> children() const;

    virtual Entry insert(const Path& path);

    virtual void insert(const Path& path, const Entry& v);

    virtual Entry find(const Path& path) const;

    virtual void remove(const Path& path);

    virtual void merge(const EntryObject&);

    virtual void patch(const EntryObject&);

    virtual void update(const EntryObject&);

    virtual void emplace(const std::string& key, Entry&&);

    template <typename V, typename... Args>
    void emplace(const std::string& key, Args&&... args)
    {
        emplace(key, Entry(std::in_place_type_t<V>(), std::forward<Args>(args)...));
    }
};

class EntryArray
{
    std::vector<Entry> m_container_;

public:
    EntryArray() {}

    EntryArray(const EntryArray& other) : m_container_(other.m_container_) {}

    EntryArray(EntryArray&& other) : m_container_(std::move(other.m_container_)) {}

    virtual ~EntryArray() = default;

    void swap(EntryArray& other) { m_container_.swap(other.m_container_); }

    std::unique_ptr<EntryArray> copy() const;

    EntryArray& operator=(const EntryArray& other)
    {
        EntryArray(other).swap(*this);
        return *this;
    }
    //-------------------------------------------------------------------------------

    virtual Cursor<Entry> children();

    virtual Cursor<const Entry> children() const;

    virtual void clear();

    virtual void resize(std::size_t num);

    virtual size_t size() const;

    virtual Entry at(int idx);

    virtual const Entry at(int idx) const;

    virtual Entry slice(int start, int stop, int step);

    virtual const Entry slice(int start, int stop, int step) const;

    virtual Entry insert(const Path& path);

    virtual Entry find(const Path& path) const;

    virtual Entry push_back();

    virtual Entry pop_back();

    virtual void push_back(const Entry& v);

    virtual void emplace_back(Entry&&);

    template <typename V, typename... Args>
    void emplace(Args&&... args)
    {
        emplace_back(Entry(std::in_place_type_t<V>(), std::forward<Args>(args)...));
    }
};

class Entry : public entry_value_union
{

public:
    typedef entry_value_union value_type;

    typedef traits::type_tags<value_type> value_type_tags;

    typedef value_type base_type;

    template <typename... Args>
    Entry(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    Entry(const Entry& other) : base_type(other) {}

    Entry(Entry&& other) : base_type(std::move(other)) {}

    ~Entry() = default;

    void swap(Entry& other) { base_type::swap(other); }

    Entry& operator=(const Entry& other)
    {
        Entry(other).swap(*this);
        return *this;
    }

    template <typename V>
    Entry& operator=(const V& v)
    {
        set_value<V>(v);
        return *this;
    }

    Entry& operator=(const char* v)
    {
        set_value<std::string>(v);
        return *this;
    }

    //-------------------------------------------------------------------------

    std::size_t type() const { return base_type::index(); }

    bool is_null() const { return type() == value_type_tags::Null; }

    void clear() { base_type::emplace<std::nullptr_t>(nullptr); }

    bool empty() const { return size() == 0; }

    size_t size() const;

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    //-------------------------------------------------------------------------

    void set_value(Entry&&);

    void set_value(const Entry& v) { set_value(Entry(v)); }

    template <typename V, typename... Args>
    void set_value(Args&&... args) { set_value(Entry(std::in_place_type_t<V>(), std::forward<Args>(args)...)); }

    const Entry get_value() const;

    template <typename V, typename... Args>
    void as(Args&&... args) { set_value(Entry(std::in_place_type_t<V>(), std::forward<Args>(args)...)); }

    template <typename V>
    V as() const { return std::get<V>(get_value()); }

    //-------------------------------------------------------------------------
    // as object

    EntryObject& as_object();

    const EntryObject& as_object() const;

    Entry insert(const Path& path);

    template <typename V, typename... Args>
    void emplace(const std::string& key, Args&&... args) { as_object().template emplace<V>(key, std::forward<Args>(args)...); }

    template <typename... Args>
    Entry find(Args&&... args) const { as_object().find(std::forward<Args>(args)...); }

    //-------------------------------------------------------------------------
    // as array
    EntryArray& as_array();

    const EntryArray& as_array() const;

    void resize(std::size_t num);

    Entry pop_back();

    template <typename... Args>
    void push_back(Args&&... args) { as_array().push_back(std::forward<Args>(args)...); }

    template <typename V, typename... Args>
    void emplace_back(Args&&... args) { as_array().emplace<V>(std::forward<Args>(args)...); }

    //-------------------------------------------------------------------------
    // access

    Entry operator[](const char* path) { return operator[](std::string(path)); }
    const Entry operator[](const char* path) const { return operator[](std::string(path)); }

    Entry operator[](const std::string& path);
    const Entry operator[](const std::string& path) const;

    Entry operator[](int idx);
    const Entry operator[](int idx) const;

    Entry slice(int start, int stop, int step = 1);
    const Entry slice(int start, int stop, int step = 1) const;

    Entry operator[](const Path& path);
    const Entry operator[](const Path& path) const;
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_