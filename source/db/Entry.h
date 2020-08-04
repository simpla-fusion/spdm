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
    Entry* m_self_;

public:
    EntryObject(Entry* self);

    virtual ~EntryObject();

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    static std::shared_ptr<EntryObject> create(Entry* self, const std::string& request = "");

    static bool add_creator(const std::string& c_id, const std::function<EntryObject*()>&);

    Entry* self() const { return m_self_; }

    void self(Entry* s) { m_self_ = s; }

    //-------------------------------------------------------------------------------

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
    Entry* m_self_;

public:
    EntryArray(Entry* self);
    virtual ~EntryArray();

    EntryArray(const EntryArray&) = delete;

    EntryArray(EntryArray&&) = delete;

    static std::shared_ptr<EntryArray> create(Entry*, const std::string& request = "");

    virtual std::shared_ptr<EntryArray> copy() const = 0;

    Entry* self() const { return m_self_; }

    void self(Entry* s) { m_self_ = s; }

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
public:
    typedef entry_base base_type;
    typedef traits::type_tags<entry_base> type_tags;

    Entry();
    virtual ~Entry();
    Entry(const Entry& other) : base_type(other) {}
    Entry(Entry&& other) : base_type(std::move(other)) {}

    void swap(Entry& other)
    {
        base_type::swap(other);

        bind_self();
        other.bind_self();
    }

    void bind_self()
    {
        if (type() == type_tags::Object)
        {
            std::get<type_tags::Object>(*this)->self(this);
        }
        else if (type() == type_tags::Array)
        {
            std::get<type_tags::Array>(*this)->self(this);
        }
    }

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
    // for reference

    Entry& fetch();

    const Entry& fetch() const;

    void update();

    std::size_t type() const;

    void clear();

    //---------------------------------------------------------------------

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    template <typename V>
    void as(const V& v)
    {
        fetch().emplace<V>(v);
        update();
    }

    template <typename V>
    void as(V&& v)
    {
        fetch().emplace<V>(std::forward<V>(v));
        update();
    }

    template <typename V>
    V& as() { return std::get<V>(fetch()); }

    template <typename V>
    const V& as() const { return std::get<V>(fetch()); }

    std::shared_ptr<DataBlock> as_block();
    std::shared_ptr<const DataBlock> as_block() const;

    std::shared_ptr<EntryObject> as_object();
    std::shared_ptr<const EntryObject> as_object() const;

    std::shared_ptr<EntryArray> as_array();
    std::shared_ptr<const EntryArray> as_array() const;

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