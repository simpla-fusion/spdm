#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
#include "../utility/TypeTraits.h"
#include "Cursor.h"
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

    virtual ~EntryObject();

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    //-------------------------------------------------------------------------------
    Entry* holder() const { return m_holder_; }

    virtual std::unique_ptr<EntryObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    //------------------------------------------------------------------

    virtual Cursor<Entry> children() = 0;

    virtual Cursor<const Entry> children() const = 0;

    virtual void insert(const std::string& path, const Entry&) = 0;

    virtual Entry fetch(const std::string& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual Cursor<Entry> select(const XPath& path) = 0;

    virtual Cursor<const Entry> select(const XPath& path) const = 0;
};

class EntryArray
{
    Entry* m_holder_;
    std::vector<Entry> m_container_;

public:
    EntryArray(Entry* holder) : m_holder_(holder) {}

    EntryArray(const EntryArray&);

    EntryArray(EntryArray&&);

    ~EntryArray() = default;
    //-------------------------------------------------------------------------------
    Entry* holder() const { return m_holder_; }

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    void resize(std::size_t num);

    void clear() { m_container_.clear(); }

    size_t size() const { return m_container_.size(); }

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
    std::weak_ptr<EntryObject> m_parent_;

public:
    typedef entry_base base_type;
    typedef traits::type_tags<entry_base> type_tags;

    Entry();

    Entry(const XPath& v);

    template <typename V>
    Entry(V const& v) { emplace<V>(v); }

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

    std::shared_ptr<EntryObject> parent() const { return m_parent_.lock(); }

    bool is_root() const { return m_parent_.expired(); }

    //---------------------------------------------------------------------
    std::size_t type() const { return index(); }

    void clear() { base_type::emplace<std::nullptr_t>(nullptr); }

    std::size_t size() const;

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    template <typename V>
    void as(const V& v) { self().emplace<V>(v); }

    template <typename V>
    void as(V&& v) { self().emplace<V>(std::forward<V>(v)); }

    template <typename V>
    V& as() { return std::get<V>(self()); }

    template <typename V>
    const V& as() const { return std::get<V>(self()); }

    DataBlock& as_block();
    const DataBlock& as_block() const;

    EntryObject& as_object();
    const EntryObject& as_object() const;

    EntryArray& as_array();

    const EntryArray& as_array() const;

    void resize(std::size_t num);

    template <typename V>
    void push_back(V&& v) { as_array().push_back().emplace<V>(v); }

    Entry pop_back();

    Entry operator[](int idx);

    Entry operator[](int idx) const;

    Entry operator[](const XPath& path);

    Entry operator[](const XPath& path) const;
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_