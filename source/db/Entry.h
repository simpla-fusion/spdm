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
M_REGISITER_TYPE_TAG(Reference, sp::db::EntryReference);

namespace sp::db
{

class EntryObject : public std::enable_shared_from_this<EntryObject>
{
    std::weak_ptr<EntryObject> m_parent_;
    std::string m_name_;

public:
    EntryObject();

    virtual ~EntryObject();

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    //-------------------------------------------------------------------------------
    std::shared_ptr<EntryObject> parent() const { return m_parent_.lock(); }

    void parent(std::shared_ptr<EntryObject> p) { m_parent_ = p; }

    std::string name() const { return m_name_; }

    virtual std::unique_ptr<EntryObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    //------------------------------------------------------------------

    virtual Cursor<Entry> select(const XPath& path) = 0;

    virtual Cursor<const Entry> select(const XPath& path) const = 0;

    virtual Cursor<Entry> children() = 0;

    virtual Cursor<const Entry> children() const = 0;

    virtual Cursor<std::pair<const std::string, Entry>> kv_items() = 0;

    virtual Cursor<const std::pair<const std::string, Entry>> kv_items() const = 0;

    //------------------------------------------------------------------

    virtual Entry insert(const std::string& path) = 0;

    virtual Entry insert(const XPath& path) = 0;

    virtual const Entry get(const std::string& path) const = 0;

    virtual const Entry get(const XPath& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual void erase(const XPath& path) = 0;

    Entry operator[](const std::string& path) { return insert(path); }

    Entry operator[](const XPath& path) { return insert(path); }

    const Entry operator[](const std::string& path) const { return get(path); }

    const Entry operator[](const XPath& path) const { return get(path); }
};

class EntryReference
{
    std::shared_ptr<EntryObject> m_root_;
    XPath m_path_;

public:
    EntryReference(std::shared_ptr<EntryObject> root, XPath const& path) : m_root_(root), m_path_(path) {}
    EntryReference(const EntryReference& other) : m_root_(other.m_root_), m_path_(other.m_path_) {}
    EntryReference(EntryReference&& other) : m_root_(other.m_root_), m_path_(other.m_path_) {}
    ~EntryReference() = default;

    void swap(EntryReference& other)
    {
        std::swap(m_root_, other.m_root_);
        std::swap(m_path_, other.m_path_);
    }

    Entry fetch();
    void push(Entry const&);
};

class EntryArray
{
    std::weak_ptr<EntryObject> m_parent_;
    std::string m_name_;
    std::vector<Entry> m_container_;

public:
    EntryArray(std::weak_ptr<EntryObject> parent) : m_parent_(parent) {}

    ~EntryArray() = default;

    EntryArray(const EntryArray&) = delete;

    EntryArray(EntryArray&&) = delete;

    //-------------------------------------------------------------------------------
    std::shared_ptr<EntryObject> parent() { return m_parent_.lock(); }

    void parent(std::shared_ptr<EntryObject> p) { m_parent_ = p; }

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    void resize(std::size_t num);

    void clear() { m_container_.clear(); }

    size_t size() const { return m_container_.size(); }

    //-------------------------------------------------------------------------------

    Entry& push_back();

    Entry pop_back();

    Entry& get(int idx) { return m_container_[idx]; }

    const Entry& get(int idx) const { return m_container_[idx]; }

    Entry& operator[](int idx) { return m_container_[idx]; }

    const Entry& operator[](int idx) const { return m_container_[idx]; }
};

typedef std::variant<std::nullptr_t,
                     std::shared_ptr<EntryObject>,
                     EntryArray,
                     EntryReference,                     //Reference
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

public:
    typedef entry_base base_type;
    typedef traits::type_tags<entry_base> type_tags;

    Entry();
    Entry(std::weak_ptr<EntryObject> parent);
    ~Entry();
    Entry(const Entry& other);
    Entry(Entry&& other);

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

    bool is_root() const;

    //---------------------------------------------------------------------
    // for reference

    // Entry& self();

    // const Entry& self() const;

    // void fetch(const std::string& request);

    // void fetch(const XPath& request);

    // std::shared_ptr<EntryObject> parent() const { return m_parent_.lock(); }

    // void parent(std::weak_ptr<EntryObject> p) { m_parent_ = p; }

    // Cursor<Entry> children();

    // Cursor<const Entry> children() const;

    std::size_t type() const;

    void clear() { base_type::emplace<std::nullptr_t>(nullptr); }

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

    Entry push_back();

    Entry pop_back();

    Entry operator[](int idx);

    Entry operator[](int idx) const;

    Entry operator[](const XPath& path);

    Entry operator[](const XPath& path) const;
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_