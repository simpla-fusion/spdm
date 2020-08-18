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
class EntryObject;
class EntryArray;
class EntryReference;
class DataBlock;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, sp::db::EntryArray);
M_REGISITER_TYPE_TAG(Block, sp::db::DataBlock);
M_REGISITER_TYPE_TAG(Reference, sp::db::XPath);

namespace sp::db
{

class EntryObject
    : public std::enable_shared_from_this<EntryObject>
{
    Entry* m_holder_;

public:
    EntryObject(Entry* holder = nullptr);

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    virtual ~EntryObject() = default;

    //-------------------------------------------------------------------------------
    Entry* holder() const;

    virtual std::unique_ptr<EntryObject> copy() const;

    virtual size_t size() const;

    virtual void clear();

    virtual Cursor<Entry> children();

    virtual Cursor<const Entry> children() const;

    virtual Cursor<std::pair<const std::string, Entry>> kv_items();

    virtual Cursor<const std::pair<const std::string, Entry>> kv_items() const;

    virtual void insert(const XPath& path, const Entry&);

    virtual Entry query(const XPath& path) const;

    virtual void remove(const XPath& path);

    virtual void update(const EntryObject& patch);

    Entry operator[](const XPath& path);

    const Entry operator[](const XPath& path) const;
};
class EntryArray
{
    Entry* m_holder_;
    std::vector<Entry> m_container_;

public:
    EntryArray(Entry* holder) : m_holder_(holder) {}

    EntryArray(const EntryArray& other) : m_holder_(nullptr), m_container_(other.m_container_) {}

    EntryArray(EntryArray&& other) : m_holder_(nullptr), m_container_(std::move(other.m_container_)) {}

    ~EntryArray() = default;

    void swap(EntryArray& other) { m_container_.swap(other.m_container_); }

    EntryArray& operator=(const EntryArray& other)
    {
        EntryArray(other).swap(*this);
        return *this;
    }
    //-------------------------------------------------------------------------------

    Entry* holder() const { return m_holder_; }

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    void resize(std::size_t num);

    void clear();

    size_t size() const;

    void insert(const XPath& path, const Entry&);

    Entry query(const XPath& path) const;

    void remove(const XPath& path);

    void update(const EntryArray& patch);

    Entry& push_back();

    Entry pop_back();

    Entry& at(int idx) { return m_container_.at(idx); }

    const Entry& at(int idx) const { return m_container_.at(idx); }

    Entry& operator[](int idx) { return m_container_.at(idx); }

    const Entry& operator[](int idx) const { return m_container_.at(idx); }
};

typedef std::variant<std::nullptr_t,
                     std::shared_ptr<EntryObject>,
                     EntryArray,
                     XPath,                              //Reference
                     DataBlock,                          //Block
                     std::string,                        //String,
                     bool,                               //Boolean,
                     int,                                //Integer,
                     long,                               //Long,
                     float,                              //Float,
                     double,                             //Double,
                     std::complex<double>,               //Complex,
                     std::array<int, 3>,                 //IntVec3,
                     std::array<long, 3>,                //LongVec3,
                     std::array<float, 3>,               //FloatVec3,
                     std::array<double, 3>,              //DoubleVec3,
                     std::array<std::complex<double>, 3> //ComplexVec3,
                     >
    entry_base;

class Entry : public entry_base
{
    Entry* m_parent_;

public:
    typedef entry_base base_type;

    typedef traits::type_tags<entry_base> type_tags;

    Entry(Entry* parent = nullptr);

    Entry(Entry* parent, const XPath&);

    Entry(const Entry& other);

    Entry(Entry&& other);

    ~Entry();

    void swap(Entry& other);

    Entry& operator=(const Entry& other)
    {
        Entry(other).swap(*this);
        return *this;
    }

    template <typename V>
    Entry& operator=(const V& v)
    {
        as<V>(v);
        return *this;
    }

    Entry& operator=(const char* v)
    {
        as<std::string>(v);
        return *this;
    }

    Entry* parent() const { return m_parent_; }

    bool is_root() const { return m_parent_ == nullptr; }

    //-------------------------------------------------------------------------

    std::size_t type() const { return base_type::index(); }

    bool empty() const { return base_type::index() == type_tags::Null; }

    void clear() { base_type::emplace<std::nullptr_t>(nullptr); }

    //-------------------------------------------------------------------------
    // CRUD operation

    Entry insert(const XPath& path);

    void insert(const XPath& path, const Entry&);

    Entry query(const XPath& path = XPath{}) const;

    void remove(const XPath&);

    void update(const Entry&);

    void update(Entry&&);

    //-------------------------------------------------------------------------

    template <typename V, typename... Args>
    void as(Args&&... args)
    {
        Entry v;
        v.emplace<V>(std::forward<Args>(args)...);
        update(std::move(v));
    }

    template <typename V>
    V as() const { return std::get<V>(query()); }

    std::size_t size() const;

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    EntryObject& as_object();
    const EntryObject& as_object() const;

    EntryArray& as_array();
    const EntryArray& as_array() const;

    Entry operator[](const XPath& path) const;

    //-------------------------------------------------------------------------

    void resize(std::size_t num);

    Entry push_back();

    template <typename V, typename... Args>
    void emplace_back(Args&&... args) { as_array().push_back().as<V>(std::forward<Args>(args)...); }

    Entry pop_back();
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_