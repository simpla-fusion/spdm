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

// typedef std::pair<std::shared_ptr<EntryContainer>, Path> EntryReference;
// typedef std::pair<const std::string, Entry> EntryItem;

} // namespace sp::db

M_REGISITER_TYPE_TAG(Object, std::shared_ptr<sp::db::EntryObject>);
M_REGISITER_TYPE_TAG(Array, std::shared_ptr<sp::db::EntryArray>);
M_REGISITER_TYPE_TAG(Block, std::shared_ptr<sp::db::DataBlock>);

namespace sp::db
{
typedef std::variant<std::nullptr_t,
                     std::shared_ptr<EntryObject>,       //Object
                     std::shared_ptr<EntryArray>,        //Array
                     std::shared_ptr<DataBlock>,         //Block
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
    virtual ~EntryContainer() = default;
    EntryContainer(EntryContainer const&) = delete;
    EntryContainer(EntryContainer&&) = delete;

    virtual std::unique_ptr<EntryContainer> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;
    //-------------------------------------------------------------------------------------------------------------

    virtual Entry at(const Path& path) = 0;

    virtual Entry at(const Path& path) const = 0;

    virtual Cursor<entry_value_type> children() = 0;

    virtual Cursor<const entry_value_type> children() const = 0;

    virtual void for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) = 0;

    virtual void for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const = 0;

    //-------------------------------------------------------------------------------------------------------------

    virtual EntryContainer* sub_container(const Path::Segment& key) = 0;

    virtual const EntryContainer* sub_container(const Path::Segment& key) const = 0;

    virtual void set_value(const Path::Segment& key, entry_value_type&& v = {}) = 0;

    virtual entry_value_type get_value(const Path::Segment& key) const = 0;

    virtual void remove(const Path::Segment& path) = 0;

    //-------------------------------------------------------------------------------------------------------------

    virtual Entry insert(const Path& path);

    virtual const entry_value_type find(const Path& path) const;

    virtual void remove(const Path& path);
};

class EntryObject : public EntryContainer
{

public:
    using EntryContainer::find;
    using EntryContainer::insert;

    friend class Entry;

    EntryObject() = default;
    virtual ~EntryObject() = default;
    EntryObject(const EntryObject&) = delete;
    EntryObject(EntryObject&&) = delete;

    virtual size_t size() const override;

    virtual void clear() override;
    //-------------------------------------------------------------------------------------------------------------
    virtual Entry at(const Path& path) override;

    virtual Entry at(const Path& path) const override;

    virtual Cursor<entry_value_type> children() override;

    virtual Cursor<const entry_value_type> children() const override;

    virtual void for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) override;

    virtual void for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const override;

    //---------------------------------------------------------------------------------

    virtual EntryContainer* sub_container(const Path::Segment& key) override;

    virtual const EntryContainer* sub_container(const Path::Segment& key) const override;

    virtual void set_value(const Path::Segment& key, entry_value_type&& v = {}) override;

    virtual entry_value_type get_value(const Path::Segment& key) const override;

    //---------------------------------------------------------------------------------

    virtual void remove(const Path::Segment& path) override;

    virtual void merge(const EntryObject&) = 0;

    virtual void patch(const EntryObject&) = 0;

    virtual void update(const EntryObject&) = 0;
};

class EntryArray : public EntryContainer
{
    std::vector<entry_value_type> m_container_;

public:
    friend class Entry;
    using EntryContainer::find;
    using EntryContainer::insert;

    EntryArray() = default;

    virtual ~EntryArray() = default;

    EntryArray(const EntryArray& other) : m_container_(other.m_container_) {}

    EntryArray(EntryArray&& other) : m_container_(std::move(other.m_container_)) {}

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

    virtual void resize(std::size_t num);

    virtual Cursor<entry_value_type> children() override;

    virtual Cursor<const entry_value_type> children() const override;

    virtual void for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) override;

    virtual void for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const override;

    virtual Entry at(const Path& path) override;

    virtual Entry at(const Path& path) const override;

    virtual Entry pop_back();

    virtual Entry push_back();
    //-------------------------------------------------------------------------------
    virtual void remove(const Path::Segment& key) override;

    virtual EntryContainer* sub_container(const Path::Segment& key) override;

    virtual const EntryContainer* sub_container(const Path::Segment& key) const override;

    virtual void set_value(const Path::Segment& key, entry_value_type&& v = {}) override;

    virtual entry_value_type get_value(const Path::Segment& key) const override;

    virtual entry_value_type get_value(int idx);

    virtual entry_value_type get_value(int idx) const;

    virtual entry_value_type slice(int start, int stop, int step);

    virtual entry_value_type slice(int start, int stop, int step) const;
};

class Entry
{
    entry_value_type m_value_;
    Path m_path_;

public:
    typedef entry_value_type value_type;

    typedef entry_value_type_tags value_type_tags;

    Entry();

    template <typename T>
    Entry(const T& v, Path const& p) : m_value_(v), m_path_(p) {}

    Entry(const entry_value_type& v, Path const& p = {}) : m_value_(v), m_path_(p) {}

    Entry(const Entry& other) : m_value_(other.m_value_), m_path_(other.m_path_) {}

    Entry(Entry&& other) : m_value_(std::move(other.m_value_)), m_path_(std::move(other.m_path_)) {}

    ~Entry() = default;

    // template <typename... Args>
    // Entry(Args&&... args) : m_value_(std::forward<Args>(args)...) {}

    Entry(const value_type& v) : m_value_(v) {}

    Entry(value_type&& v) : m_value_(std::forward<value_type>(v)) {}

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

    //-------------------------------------------------------------------------

    std::size_t type() const { return m_value_.index(); }

    bool is_null() const { return type() == value_type_tags::Null; }

    void clear() { m_value_.emplace<std::nullptr_t>(nullptr); }

    bool empty() const { return size() == 0; }

    size_t size() const;

    void resize(std::size_t num);

    //-------------------------------------------------------------------------

    Cursor<entry_value_type> children();

    Cursor<const entry_value_type> children() const;

    void for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&);

    void for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const;

    Entry at(const Path&);

    const Entry at(const Path&) const;

    //-------------------------------------------------------------------------

    void set_value(value_type&& v);

    void set_value(const value_type& v) { set_value(entry_value_type(v)); }

    entry_value_type get_value() const;

    template <typename V, typename... Args>
    void as(Args&&... args) { set_value(value_type(std::in_place_type_t<V>(), std::forward<Args>(args)...)); }

    template <typename V>
    V as() const { return std::get<V>(get_value()); }

    //-------------------------------------------------------------------------
    // as object

    EntryObject& as_object();

    const EntryObject& as_object() const;

    //-------------------------------------------------------------------------
    // as array
    EntryArray& as_array();

    const EntryArray& as_array() const;

    Entry pop_back();

    Entry push_back();

    //-------------------------------------------------------------------------
    // access

    // Entry at(const Path::Segment&);
    // const Entry at(const Path::Segment&) const;

    template <typename T>
    inline Entry operator[](const T& idx) { return at(Path(idx)); }
    template <typename T>
    inline const Entry operator[](const T& idx) const { return at(Path(idx)); }

    inline Entry slice(int start, int stop, int step = 1) { return at(Path(start, stop, step)); }
    inline const Entry slice(int start, int stop, int step = 1) const { return at(Path(start, stop, step)); }

    inline Entry operator[](const Path& path);
    inline const Entry operator[](const Path& path) const;
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

namespace literals
{
using namespace std::complex_literals;
using namespace std::string_literals;
} // namespace literals
} // namespace sp::db

#endif //SP_ENTRY_H_