#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Iterator.h"
#include "Range.h"
#include <any>
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <ostream>
#include <variant>

namespace sp
{
struct XPath;

class EntryInterface;

class Entry
{
private:
    std::string m_prefix_;
    std::shared_ptr<EntryInterface> m_pimpl_;

    std::shared_ptr<EntryInterface> get_self() const;
    std::shared_ptr<EntryInterface> get_self();

public:
    enum Type
    {
        Null = 0,
        Single = 1,
        Tensor = 2,
        Block = 3,
        Array = 4,
        Object = 5
    };

    typedef std::variant<std::string,
                         bool, int, double,
                         std::complex<double>,
                         std::array<int, 3>,
                         std::array<double, 3>>
        single_t;

    typedef std::tuple<std::shared_ptr<void> /* data ponter*/,
                       const std::type_info& /* type information */,
                       std::vector<size_t> /* dimensions */>
        tensor_t;

    typedef std::tuple<std::shared_ptr<void> /* data ponter*/,
                       std::any /* type description*/,
                       std::vector<size_t> /* shapes */,
                       std::vector<size_t> /* offset */,
                       std::vector<size_t> /* strides */,
                       std::vector<size_t> /* dimensions */
                       >
        block_t;

    friend class EntryInterface;

    typedef Entry this_type;

    Entry();

    explicit Entry(const std::string& uri);

    explicit Entry(const std::shared_ptr<EntryInterface>& p,
                   const std::string& prefix = "");

    explicit Entry(EntryInterface* p,
                   const std::string& prefix = "");

    Entry(const this_type&);

    Entry(this_type&&);

    ~Entry();

    void swap(this_type&);

    this_type& operator=(this_type const& other);

    bool operator==(this_type const& other) const;

    operator bool() const { return !is_null(); }

    void resolve();

    // metadata
    Type type() const;
    bool is_null() const;
    bool is_single() const;
    bool is_tensor() const;
    bool is_block() const;
    bool is_array() const;
    bool is_object() const;

    bool is_root() const;
    bool is_leaf() const;

    //

    std::string full_path() const;

    std::string relative_path() const;

    // attributes

    bool has_attribute(const std::string& name) const;

    const single_t get_attribute_raw(const std::string& name) const;

    void set_attribute_raw(const std::string& name, const single_t& value);

    void remove_attribute(const std::string& name);

    template <typename V>
    const single_t get_attribute(const std::string& name)
    {
        return std::get<V>(get_attribute_raw(name));
    };

    void set_attribute(const std::string& name, const char* value)
    {
        set_attribute_raw(name, single_t{std::string(value)});
    }

    template <typename V>
    void set_attribute(const std::string& name, const V& value)
    {
        set_attribute_raw(name, single_t{value});
    }

    std::map<std::string, single_t> attributes() const;

    //----------------------------------------------------------------------------------
    // level 0
    //
    // as leaf

    void set_single(const single_t&);

    single_t get_single() const;

    template <typename V>
    void set_value(const V& v) { set_single(single_t(v)); };

    template <typename V>
    V get_value() const { return std::get<V>(get_single()); }

    void set_tensor(const tensor_t&);

    tensor_t get_tensor() const;

    void set_block(const block_t&);

    block_t get_block() const;

    template <typename... Args>
    void set_block(Args&&... args) { return selt_block(std::make_tuple(std::forward<Args>(args)...)); };

    // as Tree
    // as container

    const Entry& self() const;

    Entry& self();

    Entry parent() const;

    Range<Entry> children() const;

    void clear();

    // as array

    Entry operator[](int); // access  specified child

    Entry operator[](int) const; // access  specified child

    Entry push_back(); // append new item

    Entry pop_back(); // remove and return last item

    // as object
    // @note : map is unordered

    Entry insert(const std::string& key); // if key is not exists then insert node at key else return entry at key

    bool has_a(const std::string& key) const;

    Entry find(const std::string& key) const;

    Entry operator[](const char* c) const { return operator[](std::string(c)); }

    Entry operator[](const char* c) { return operator[](std::string(c)); }

    Entry operator[](const std::string&) const; // access  specified child

    Entry operator[](const std::string&); // access or insert specified child

    void remove(const std::string&);

    //-------------------------------------------------------------------
    // level 1
    // xpath

    Entry insert(const XPath&);

    Range<Entry> find(const XPath&) const;

    typedef std::function<bool(const Entry&)> pred_fun;

    Range<Entry> find(const pred_fun&) const;

    int update(const Range<std::string, Entry>&, const Entry&);

    int remove(const Range<std::string, Entry>&);

    //-------------------------------------------------------------------
    // level 2

    size_t depth() const; // parent.depth +1

    size_t height() const; // max(children.height) +1

    Range<Entry> slibings() const; // return slibings

    Range<Entry> ancestor() const; // return ancestor

    Range<Entry> descendants() const; // return descendants

    Range<Entry> leaves() const; // return leave nodes in traversal order

    Range<Entry> shortest_path(Entry const& target) const; // return the shortest path to target

    ptrdiff_t distance(const this_type& target) const; // lenght of shortest path to target
};

std::string to_string(const Entry::single_t& s);

Entry::single_t from_string(const std::string& s);

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp

#endif // SP_ENTRY_H_
