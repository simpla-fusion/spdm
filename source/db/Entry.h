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

class EntryObject : public std::enable_shared_from_this<EntryObject>
{

public:
    friend class Entry;

    EntryObject() = default;
    virtual ~EntryObject() = default;
    EntryObject(const EntryObject&) = delete;
    EntryObject(EntryObject&&) = delete;

    virtual std::unique_ptr<EntryObject> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;
    //-------------------------------------------------------------------------------------------------------------
    // as container

    virtual Entry at(const Path& path) = 0;

    virtual Entry at(const Path& path) const = 0;

    virtual Cursor<entry_value_type> children() = 0;

    virtual Cursor<const entry_value_type> children() const = 0;

    virtual void for_each(std::function<void(const std::string&, entry_value_type&)> const&) = 0;

    virtual void for_each(std::function<void(const std::string&, const entry_value_type&)> const&) const = 0;

    // access children

    virtual Entry insert(const std::string&, entry_value_type&&) = 0;

    virtual void set_value(const std::string& key, entry_value_type&& v = {}) = 0;

    virtual entry_value_type get_value(const std::string& key) const = 0;

    virtual void remove(const std::string& path) = 0;

    //------------------------------------------------------------------------------
    // fundamental operation ：
    /**
     *  Create 
     */
    virtual Entry insert(const Path&, entry_value_type&&);
    /**
     * Modify
     */
    virtual void update(const Path&, entry_value_type&& v = {});
    /**
     * Retrieve
     */
    virtual entry_value_type find(const Path& key) const;

    /**
     *  Delete 
     */
    virtual void remove(const Path& path);

    //------------------------------------------------------------------------------
    // advanced extension functions
    virtual void merge(const EntryObject&);

    virtual void patch(const EntryObject&);

    virtual void update(const EntryObject&);
};

class EntryArray : public std::enable_shared_from_this<EntryArray>
{
    std::vector<entry_value_type> m_container_;

public:
    friend class Entry;

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

    virtual std::unique_ptr<EntryArray> copy() const { return std::unique_ptr<EntryArray>(new EntryArray(*this)); };

    virtual void clear();

    virtual size_t size() const;

    virtual void resize(std::size_t num);

    virtual Cursor<entry_value_type> children();

    virtual Cursor<const entry_value_type> children() const;

    virtual void for_each(std::function<void(int, entry_value_type&)> const&);

    virtual void for_each(std::function<void(int, const entry_value_type&)> const&) const;

    virtual Entry at(const Path& path);

    virtual Entry at(const Path& path) const;

    virtual Entry pop_back();

    virtual Entry push_back();

    virtual entry_value_type slice(int start, int stop, int step);

    virtual entry_value_type slice(int start, int stop, int step) const;

    virtual void set_value(int idx, entry_value_type&&);

    virtual entry_value_type get_value(int idx) const;

    //------------------------------------------------------------------------------
    // CRUD operation

    /**
     *  Create 
    */
    virtual Entry insert(const Path&, entry_value_type&&);
    /**
     * Modify
    */
    virtual void update(const Path&, entry_value_type&& v = {});
    /**
     * Retrieve
     */
    virtual entry_value_type find(const Path& key) const;

    /**
     *  Delete 
    */
    virtual void remove(const Path& path);
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

    bool empty() const { return size() == 0; }

    size_t size() const;

    void resize(std::size_t num);

    void clear() { m_value_.emplace<std::nullptr_t>(nullptr); }

    entry_value_type& value() { return m_value_; }
    const entry_value_type& value() const { return m_value_; }

    Path& path() { return m_path_; }
    const Path& path() const { return m_path_; }
    //-------------------------------------------------------------------------

    Cursor<entry_value_type> children();

    Cursor<const entry_value_type> children() const;

    void for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&);

    void for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const;

    //-------------------------------------------------------------------------

    void set_value(value_type&& v);

    template <typename V, typename... Args>
    void set_value(Args&&... args) { set_value(value_type(std::in_place_type_t<V>(), std::forward<Args>(args)...)); }

    template <typename V, typename... Args>
    void as(Args&&... args) { set_value<V>(std::forward<Args>(args)...); }

    entry_value_type get_value() const;

    template <typename V>
    V get_value() const { return std::get<V>(get_value()); }

    template <typename V>
    V as() const { return std::get<V>(get_value()); }

    //-------------------------------------------------------------------------
    // as object

    std::shared_ptr<EntryObject> as_object();
    std::shared_ptr<const EntryObject> as_object() const;

    std::shared_ptr<EntryArray> as_array();
    std::shared_ptr<const EntryArray> as_array() const;

    //-------------------------------------------------------------------------
    // access

    Entry at(const Path&);
    const Entry at(const Path&) const;

    Entry pop_back();
    Entry push_back();

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