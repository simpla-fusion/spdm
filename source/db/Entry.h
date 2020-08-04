#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
#include "../utility/Path.h"
#include "../utility/TypeTraits.h"
#include "Cursor.h"
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
M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::EntryArray>);

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

    virtual std::shared_ptr<EntryObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    virtual std::size_t count(const std::string& name) = 0;

    virtual Entry& insert(const std::string& path) = 0;

    virtual Entry& insert(const Path& path) = 0;

    virtual const Entry& at(const std::string& path) const = 0;

    virtual const Entry& at(const Path& path) const = 0;

    virtual Cursor<Entry> find(const std::string& path) = 0;

    virtual Cursor<Entry> find(const Path& path) = 0;

    virtual Cursor<const Entry> find(const std::string& path) const = 0;

    virtual Cursor<const Entry> find(const Path& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual void erase(const Path& path) = 0;

    virtual Cursor<Entry> children() = 0;

    virtual Cursor<const Entry> children() const = 0;

    virtual Cursor<std::pair<const std::string, Entry>> kv_items() = 0;

    virtual Cursor<std::pair<const std::string, Entry>> kv_items() const = 0;

    // level 1

    virtual Cursor<Entry> select(const std::string& path) = 0;

    virtual Cursor<Entry> select(const Path& path) = 0;

    virtual Cursor<const Entry> select(const std::string& path) const = 0;

    virtual Cursor<const Entry> select(const Path& path) const = 0;

    template <typename P>
    Entry& operator[](const P& path) { return insert(path); }

    template <typename P>
    const Entry& operator[](const P& path) const { return at(path); }
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

    virtual Cursor<Entry> children() = 0;

    virtual Cursor<const Entry> children() const = 0;

    virtual size_t size() const = 0;

    virtual void resize(std::size_t num) = 0;

    virtual void clear() = 0;

    virtual Entry& push_back() = 0;

    virtual void pop_back() = 0;

    virtual Entry& at(int idx) = 0;

    virtual const Entry& at(int idx) const = 0;

    Entry& operator[](int idx) { return at(idx); }

    const Entry& operator[](int idx) const { return at(idx); }
};

typedef std::variant<std::nullptr_t,
                     std::shared_ptr<EntryObject>,
                     std::shared_ptr<EntryArray>,
                     std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
                     std::string,                                                 //String,
                     bool,                                                        //Boolean,
                     int,                                                         //Integer,
                     long,                                                        //Long,
                     float,                                                       //Float,
                     double                                                       //,      //Double,
                     // std::complex<double>,                                        //Complex,
                     // std::array<int, 3>,                                          //IntVec3,
                     // std::array<long, 3>,                                         //LongVec3,
                     // std::array<float, 3>,                                        //FloatVec3,
                     // std::array<double, 3>,                                       //DoubleVec3,
                     // std::array<std::complex<double>, 3>,                         //ComplexVec3,
                     //std::any
                     >
    entry_base;

class Entry : public entry_base
{
public:
    typedef entry_base base_type;
    typedef traits::type_tags<entry_base> type_tags;
    typedef Cursor<Entry> cursor;
    typedef Cursor<const Entry> const_cursor;

    Entry();
    ~Entry();
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

    std::size_t type() const { return base_type::index(); }

    void clear() { base_type::emplace<std::nullptr_t>(nullptr); }

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    template <typename V>
    void as(const V& v) { base_type::emplace<V>(v); }

    template <typename V>
    void as(V&& v) { base_type::emplace<V>(std::forward<V>(v)); }

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

    Entry& push_back() { return as_array().push_back(); }

    void pop_back() { as_array().pop_back(); }

    void resize(size_t num) { as_array().resize(num); }
};


std::ostream& operator<<(std::ostream& os, Entry const& entry);

// std::string to_string(Entry const& s);

// Entry from_string(const std::string& s, int idx = 0);

// template <typename TNode>
// using entry_wrapper = traits::template_copy_type_args<
//     HierarchicalTree,
//     traits::concatenate_t<
//         std::variant<TNode,
//                      EntryObject,
//                      EntryArray>,
//         element_types>>;

} // namespace sp::db

#endif //SP_ENTRY_H_