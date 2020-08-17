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
class DataBlock;
class Entry;
class EntryObject;
class EntryArray;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::EntryArray>);
M_REGISITER_TYPE_TAG(Reference, std::shared_ptr<sp::db::Entry>);
M_REGISITER_TYPE_TAG(Block, std::shared_ptr<sp::db::DataBlock>);

namespace sp::db
{

class EntryObject : std::enable_shared_from_this<EntryObject>
{
    std::weak_ptr<Entry> m_self_;

public:
    EntryObject(std::shared_ptr<Entry> self);

    virtual ~EntryObject();

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    std::shared_ptr<Entry> self() const { return m_self_.lock(); }

    void self(std::shared_ptr<Entry> s) { m_self_ = s; }

    //-------------------------------------------------------------------------------

    virtual void fetch(const XPath&){};

    virtual void update(const XPath&){};

    virtual std::shared_ptr<EntryObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    //------------------------------------------------------------------

    virtual Cursor<Entry> select(const XPath& path) = 0;

    virtual Cursor<const Entry> select(const XPath& path) const = 0;

    virtual Cursor<Entry> children() = 0;

    virtual Cursor<const Entry> children() const = 0;

    virtual Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> kv_items() = 0;

    virtual Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> kv_items() const = 0;

    //------------------------------------------------------------------

    virtual std::shared_ptr<Entry> insert(const std::string& path) = 0;

    virtual std::shared_ptr<Entry> insert(const XPath& path) = 0;

    virtual std::shared_ptr<const Entry> get(const std::string& path) const = 0;

    virtual std::shared_ptr<const Entry> get(const XPath& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual void erase(const XPath& path) = 0;

    template <typename P>
    decltype(auto) operator[](const P& path) { return *insert(path); }

    template <typename P>
    decltype(auto) operator[](const P& path) const { return *get(path); }
};

class EntryArray : std::enable_shared_from_this<EntryArray>
{
    std::weak_ptr<Entry> m_self_;

public:
    EntryArray(std::shared_ptr<Entry> self);
    virtual ~EntryArray();

    EntryArray(const EntryArray&) = delete;

    EntryArray(EntryArray&&) = delete;

    static std::shared_ptr<EntryArray> create(std::shared_ptr<Entry>, const std::string& request = "");

    virtual std::shared_ptr<EntryArray> copy() const = 0;

    std::shared_ptr<Entry> self() const { return m_self_.lock(); }

    void self(std::shared_ptr<Entry> s) { m_self_ = s; }

    //-------------------------------------------------------------------------------

    virtual Cursor<Entry> children() = 0;

    virtual Cursor<const Entry> children() const = 0;

    virtual size_t size() const = 0;

    virtual void resize(std::size_t num) = 0;

    virtual void clear() = 0;

    //-------------------------------------------------------------------------------

    virtual std::shared_ptr<Entry> push_back() = 0;

    virtual void pop_back() = 0;

    virtual std::shared_ptr<Entry> get(int idx) = 0;

    virtual std::shared_ptr<const Entry> get(int idx) const = 0;

    decltype(auto) operator[](int idx) { return *get(idx); }

    decltype(auto) operator[](int idx) const { return *get(idx); }
};

typedef std::variant<std::nullptr_t,
                     std::shared_ptr<EntryObject>,
                     std::shared_ptr<EntryArray>,
                     std::shared_ptr<Entry>,             //Reference
                     std::shared_ptr<DataBlock>,         //Block
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

class Entry : public entry_base, public std::enable_shared_from_this<Entry>
{
    std::weak_ptr<Entry> m_parent_;

public:
    typedef entry_base base_type;
    typedef traits::type_tags<entry_base> type_tags;

    Entry(std::shared_ptr<Entry> parent = nullptr);
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

    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // for reference

    Entry& self();

    const Entry& self() const;

    Entry& fetch(const std::string& request);

    Entry& fetch(const XPath& request);

    std::shared_ptr<Entry> parent() const { return m_parent_.lock(); }

    void parent(std::shared_ptr<Entry> p) { m_parent_ = p; }

    bool is_root() const { return m_parent_.expired(); }

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    std::size_t type() const { return self().index(); }

    void clear() { base_type::emplace<std::nullptr_t>(nullptr); }

    void update();

    template <typename V>
    void as(const V& v) { self().emplace<V>(v); }

    template <typename V>
    void as(V&& v) { self().emplace<V>(std::forward<V>(v)); }

    template <typename V>
    V& as() { return std::get<V>(self()); }

    template <typename V>
    const V& as() const { return std::get<V>(self()); }

    std::shared_ptr<DataBlock> as_block();
    std::shared_ptr<const DataBlock> as_block() const;

    std::shared_ptr<EntryObject> as_object();
    std::shared_ptr<const EntryObject> as_object() const;

    std::shared_ptr<EntryArray> as_array();
    std::shared_ptr<const EntryArray> as_array() const;

    // std::size_t size() const
    // {
    //     if (base_type::index() == type_tags::Array)
    //     {
    //         return as_array()->size();
    //     }
    //     else if (base_type::index() == type_tags::Object)
    //     {
    //         return as_object()->size();
    //     }
    //     else if (base_type::index() == type_tags::Reference)
    //     {
    //         return self().size();
    //     }
    //     else
    //     {
    //         return 0;
    //     }
    // }

    void resize(std::size_t num) { as_array()->resize(num); }

    decltype(auto) push_back() { return as_array()->push_back(); }

    decltype(auto) pop_back() { return as_array()->pop_back(); }

    decltype(auto) operator[](int idx) { return *(as_array()->get(idx)); }

    decltype(auto) operator[](int idx) const { return *(as_array()->get(idx)); }

    template <typename TIDX>
    decltype(auto) operator[](const TIDX& path) { return *(as_object()->insert(path)); }

    template <typename TIDX>
    decltype(auto) operator[](const TIDX& path) const { return *(as_object()->get(path)); }
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_