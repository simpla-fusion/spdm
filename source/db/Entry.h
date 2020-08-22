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
class EntryContainer;
class EntryObject;
class EntryArray;
class DataBlock;

typedef std::pair<std::shared_ptr<EntryContainer>, Path> EntryReference;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::EntryArray>);
M_REGISITER_TYPE_TAG(Block, std::shared_ptr<sp::db::DataBlock>);
M_REGISITER_TYPE_TAG(Reference, sp::db::EntryReference);
// M_REGISITER_TYPE_TAG(Item, sp::db::EntryItem)

namespace sp::db
{
typedef std::variant<std::nullptr_t,
                     std::shared_ptr<EntryObject>,
                     std::shared_ptr<EntryArray>,
                     std::shared_ptr<DataBlock>,         //Block
                     EntryReference,                     //Reference
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
                     >
    entry_value_type;
typedef traits::type_tags<entry_value_type> entry_value_type_tags;

class EntryContainer : public std::enable_shared_from_this<EntryContainer>
{
public:
    EntryContainer() = default;

    virtual ~EntryContainer();

    // virtual std::unique_ptr<EntryContainer> copy() const = 0;

    // virtual size_t size() const = 0;

    // virtual void clear() = 0;

    // virtual Cursor<Entry> children() = 0;

    // virtual Cursor<const Entry> children() const = 0;

    virtual void insert(const Path& path, entry_value_type&& v);

    virtual Entry try_insert(const Path& path, entry_value_type&& v = entry_value_type{});

    virtual const Entry find(const Path& path) const;

    virtual void remove(const Path& path);

    virtual void insert(const Path::PathSegment& key, entry_value_type&&) = 0;

    virtual const Entry find(const Path::PathSegment& key) const = 0;

    virtual void remove(const Path::PathSegment& path) = 0;
};

class EntryObject : public EntryContainer
{

public:
    using EntryContainer::find;
    using EntryContainer::insert;

    friend class Entry;

    EntryObject() = default;

    EntryObject(const EntryObject&) = delete;

    EntryObject(EntryObject&&) = delete;

    virtual ~EntryObject();

    virtual void insert(const Path::PathSegment& key, entry_value_type&&);

    virtual const Entry find(const Path::PathSegment& key) const;

    virtual void remove(const Path::PathSegment& path);

    virtual void merge(const EntryObject&) = 0;

    virtual void patch(const EntryObject&) = 0;

    virtual void update(const EntryObject&) = 0;

    template <typename V, typename... Args>
    void emplace(const std::string& key, Args&&... args)
    {
        insert(key, entry_value_type(std::in_place_type_t<V>(), std::forward<Args>(args)...));
    }
};

class EntryArray : public EntryContainer
{
    std::vector<entry_value_type> m_container_;

public:
    friend class Entry;
    using EntryContainer::find;
    using EntryContainer::insert;

    EntryArray() = default;

    EntryArray(const EntryArray& other) : m_container_(other.m_container_) {}

    EntryArray(EntryArray&& other) : m_container_(std::move(other.m_container_)) {}

    virtual ~EntryArray() = default;

    void swap(EntryArray& other) { m_container_.swap(other.m_container_); }

    EntryArray& operator=(const EntryArray& other)
    {
        EntryArray(other).swap(*this);
        return *this;
    }
    //-------------------------------------------------------------------------------

    virtual std::unique_ptr<EntryContainer> copy() const override { return std::unique_ptr<EntryContainer>(new EntryArray(*this)); };

    virtual void clear() override;

    virtual size_t size() const override;

    virtual Cursor<Entry> children() override;

    virtual Cursor<const Entry> children() const override;

    virtual void insert(const Path::PathSegment& path, entry_value_type&& v) override;

    virtual const Entry find(const Path::PathSegment& key) const override;

    virtual void remove(const Path::PathSegment& key) override;

    //-------------------------------------------------------------------------------

    virtual void resize(std::size_t num);

    virtual Entry at(int idx);

    virtual const Entry at(int idx) const;

    virtual Entry slice(int start, int stop, int step);

    virtual const Entry slice(int start, int stop, int step) const;

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

class Entry
{
public:
    typedef entry_value_type value_type;

    typedef entry_value_type_tags value_type_tags;

    Entry();

    template <typename... Args>
    Entry(Args&&... args) : m_value_(std::forward<Args>(args)...) {}

    Entry(const Entry& other) : m_value_(other.m_value_) {}

    Entry(Entry&& other) : m_value_(std::move(other.m_value_)) {}

    ~Entry() = default;

    void swap(Entry& other);

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

    std::size_t type() const { return m_value_.index(); }

    bool is_null() const { return type() == value_type_tags::Null; }

    void clear() { m_value_.emplace<std::nullptr_t>(nullptr); }

    bool empty() const { return size() == 0; }

    size_t size() const;

    Cursor<Entry> children();

    Cursor<const Entry> children() const;

    Entry insert(const Path& path);

    Entry find(const Path& path) const;

    //-------------------------------------------------------------------------
    value_type& value();

    const value_type& value() const;

    void set_value(value_type&&);

    void set_value(const value_type& v);

    template <typename V, typename... Args>
    void set_value(Args&&... args) { m_value_.emplace<V>(std::forward<Args>(args)...); }

    value_type get_value();

    value_type get_value() const;

    template <typename V, typename... Args>
    void as(Args&&... args) { set_value<V>(std::forward<Args>(args)...); }

    template <typename V>
    V as() const { return std::get<V>(get_value()); }

    //-------------------------------------------------------------------------
    // as object

    EntryObject& as_object();

    const EntryObject& as_object() const;

    template <typename V, typename... Args>
    void emplace(const std::string& key, Args&&... args) { as_object().template emplace<V>(key, std::forward<Args>(args)...); }

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

private:
    value_type m_value_;
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_