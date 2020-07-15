#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Range.h"
#include <any>
#include <memory>

namespace sp
{
enum TypeTag
{
    Null = 0000, // value is invalid
    Scalar,
    Block,
    Array, // as JSON array
    Table  // key-value, C++ map or JSON object

};

/**
 * 
 * 
 *  @ref  Virtual inheritance ,https://en.wikipedia.org/wiki/Virtual_inheritance
*/

template <TypeTag TAG>
struct EntryInterface;

struct EntryInterfaceAttributes
{
    virtual bool has_attribute(std::string const& key) const = 0;

    virtual bool check_attribute(std::string const& key, std::any const& v) const = 0;

    virtual void set_attribute(const std::string&, const std::any&) = 0;

    virtual std::any get_attribute(const std::string&) const = 0;

    virtual std::any get_attribute(std::string const& key, std::any const& default_value) = 0;

    virtual void remove_attribute(const std::string&) = 0;

    virtual Range<Iterator<std::pair<std::string, std::any>>> attributes() const = 0;

    virtual void clear_attributes() = 0;
};
struct Entry;

struct EntryInterfaceBase
{

    EntryInterfaceBase() = default;

    virtual ~EntryInterfaceBase() = default;

    virtual TypeTag type_tag() const { return TypeTag::Null; }

    virtual std::shared_ptr<Entry> create() const = 0;

    virtual std::shared_ptr<Entry> copy() const = 0;

    virtual std::shared_ptr<Entry> as_interface(TypeTag tag) = 0;

    template <TypeTag I0>
    std::shared_ptr<EntryInterface<I0>> as() { return std::dynamic_pointer_cast<EntryInterface<I0>>(as_interface(I0)); }

    virtual void resolve() = 0;
};

struct Entry
    : public virtual EntryInterfaceBase,
      public virtual EntryInterfaceAttributes
{
    typedef Iterator<Entry> iterator;
    typedef Iterator<const Entry> const_iterator;
    typedef Range<iterator> range;
    typedef Range<const_iterator> const_range;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<Entry>>> iterator_kv;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<const Entry>>> const_iterator_kv;
    typedef Range<iterator_kv> range_kv;
    typedef Range<const_iterator_kv> const_range_kv;
};

template <>
struct EntryInterface<TypeTag::Null>
    : public virtual EntryInterfaceBase
{
    TypeTag type_tag() const { return TypeTag::Null; }
};

template <>
struct EntryInterface<TypeTag::Scalar>
    : public virtual EntryInterfaceBase
{
    // as scalar
    TypeTag type_tag() const { return TypeTag::Scalar; }

    virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

    virtual void set_scalar(std::any const&) = 0;

    template <typename U, typename V>
    void set_value(V const& v) { set_scalar(std::make_any<U>(v)); }

    template <typename U>
    U get_value() const { return std::any_cast<U>(get_scalar()); }
};

template <>
struct EntryInterface<TypeTag::Block>
    : public virtual EntryInterfaceBase
{
    // as leaf node
    TypeTag type_tag() const { return TypeTag::Block; }

    virtual std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const = 0; // get block

    virtual void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                               const std::type_info& /*element type*/,
                               const std::vector<size_t>& /*dimensions*/) = 0; // set block

    template <typename V, typename... Args>
    void set_block(std::shared_ptr<V> const& d, Args... args)
    {
        set_raw_block(std::reinterpret_pointer_cast<void>(d),
                      typeid(V), std::vector<size_t>{std::forward<Args>(args)...});
    }

    template <typename V, typename... Args>
    std::tuple<std::shared_ptr<V>, std::type_info const&, std::vector<size_t>> get_block() const
    {
        auto blk = get_raw_block();
        return std::make_tuple(std::reinterpret_pointer_cast<void>(std::get<0>(blk)),
                               std::get<1>(blk), std::get<2>(blk));
    }
};

struct EntryInterfaceTree
    : public virtual EntryInterfaceBase
{
    //--------------------------------------------------------
    // as tree node

    virtual size_t size() const = 0;

    virtual Entry::range children() = 0; // reutrn list of children

    virtual Entry::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(const Entry::iterator&) = 0;

    virtual void remove_children(const Entry::range&) = 0;

    virtual Entry::iterator begin() = 0;

    virtual Entry::iterator end() = 0;

    virtual Entry::const_iterator cbegin() const = 0;

    virtual Entry::const_iterator cend() const = 0;
};

template <>
struct EntryInterface<TypeTag::Array>
    : public virtual EntryInterfaceTree
{
    // as array
    TypeTag type_tag() const { return TypeTag::Array; }

    virtual std::shared_ptr<Entry> push_back(const std::shared_ptr<Entry>& p = nullptr) = 0;

    virtual std::shared_ptr<Entry> push_back(Entry&&) = 0;

    virtual std::shared_ptr<Entry> push_back(const Entry&) = 0;

    virtual Entry::range push_back(const Entry::iterator& b, const Entry::iterator& e) = 0;

    virtual std::shared_ptr<Entry> at(int idx) = 0;

    virtual std::shared_ptr<const Entry> at(int idx) const = 0;

    Entry& operator[](size_t idx) { return *at(idx); }

    const Entry& operator[](size_t idx) const { return *at(idx); }
};

template <>
struct EntryInterface<TypeTag::Table>
    : public virtual EntryInterfaceTree
{
    // as table
    TypeTag type_tag() const { return TypeTag::Table; }

    virtual Entry::const_range_kv items() const = 0;

    virtual Entry::range_kv items() = 0;

    virtual std::shared_ptr<Entry> insert(const std::string& k, std::shared_ptr<Entry> const& node) = 0;

    virtual Entry::range_kv insert(const Entry::iterator_kv& b, const Entry::iterator_kv& e) = 0;

    virtual std::shared_ptr<Entry> at(const std::string& key) = 0;

    virtual std::shared_ptr<const Entry> at(const std::string& idx) const = 0;

    virtual std::shared_ptr<Entry> find_child(const std::string&) = 0;

    virtual std::shared_ptr<const Entry> find_child(const std::string&) const = 0;

    template <typename TI0, typename TI1>
    auto insert(TI0 const& b, TI1 const& e) { return insert(iterator_kv(b), iterator_kv(e)); }

    Entry& operator[](const std::string& path) { return *at(path); }

    const Entry& operator[](const std::string& path) const { return *at(path); }
};

template <typename Backend, TypeTag TAG>
struct EntryPolicy;

template <typename Backend>
struct EntryPolicyBase
{
    virtual std::shared_ptr<Entry> self() = 0;
    virtual std::shared_ptr<Entry> parent() = 0;
};

template <typename Backend>
struct EntryPolicyBody
    : public virtual EntryInterfaceBase,
      public virtual EntryPolicyBase<Backend>
{
    void resolve() final;
};

template <typename Backend>
struct EntryPolicyAttributes
    : public virtual EntryPolicyBase<Backend>,
      public virtual EntryInterfaceAttributes
{
    //----------------------------------------------------------------
    // attributes

    bool has_attribute(std::string const& key) const;

    bool check_attribute(std::string const& key, std::any const& v) const;

    void set_attribute(const std::string&, const std::any&);

    std::any get_attribute(const std::string&) const;

    std::any get_attribute(std::string const& key, std::any const& default_value);

    void remove_attribute(const std::string&);

    Range<Iterator<std::pair<std::string, std::any>>> attributes() const;

    void clear_attributes();
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Null>
    : public virtual EntryInterface<TypeTag::Null>,
      public virtual EntryPolicyBase<Backend>
{
    std::shared_ptr<Entry> as_interface(TypeTag tag);
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Scalar>
    : public virtual EntryInterface<TypeTag::Scalar>,
      public virtual EntryPolicyBase<Backend>
{
    std::shared_ptr<Entry> as_interface(TypeTag tag);

    //--------------------------------------------------------------------
    // as scalar
    std::any get_scalar() const;

    void set_scalar(std::any const&);
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Block>
    : public virtual EntryInterface<TypeTag::Block>,
      public virtual EntryPolicyBase<Backend>
{
    std::shared_ptr<Entry> as_interface(TypeTag tag);

    // as block
    std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const; // get block

    void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                       const std::type_info& /*element type*/,
                       const std::vector<size_t>& /*dimensions*/); // set block
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Array>
    : public virtual EntryInterface<TypeTag::Array>,
      public virtual EntryPolicyBase<Backend>
{
    std::shared_ptr<Entry> as_interface(TypeTag tag);

    size_t size() const;

    Entry::range children();

    Entry::const_range children() const;

    void clear_children();

    void remove_child(Entry::iterator const&);

    void remove_children(Entry::range const&);

    Entry::iterator begin();

    Entry::iterator end();

    Entry::const_iterator cbegin() const;

    Entry::const_iterator cend() const;

    // as array
    std::shared_ptr<Entry> push_back(const std::shared_ptr<Entry>& p = nullptr);

    std::shared_ptr<Entry> push_back(Entry&&);

    std::shared_ptr<Entry> push_back(const Entry&);

    Entry::range push_back(const Entry::iterator& b, const Entry::iterator& e);

    std::shared_ptr<Entry> at(int idx);

    std::shared_ptr<const Entry> at(int idx) const;

    std::shared_ptr<Entry> find_child(size_t);

    std::shared_ptr<const Entry> find_child(size_t) const;
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Table>
    : public virtual EntryInterface<TypeTag::Table>,
      public EntryPolicyBase<Backend>
{
    std::shared_ptr<Entry> as_interface(TypeTag tag);

    size_t size() const;

    Entry::range children();

    Entry::const_range children() const;

    void clear_children();

    void remove_child(Entry::iterator const&);

    void remove_children(Entry::range const&);

    Entry::iterator begin();

    Entry::iterator end();

    Entry::const_iterator cbegin() const;

    Entry::const_iterator cend() const;

    // as table
    Entry::const_range_kv items() const;

    Entry::range_kv items();

    std::shared_ptr<Entry> insert(const std::string& k, std::shared_ptr<Entry> const& node);

    Entry::range_kv insert(const Entry::iterator_kv& b, const Entry::iterator_kv& e);

    std::shared_ptr<Entry> at(const std::string& key);

    std::shared_ptr<const Entry> at(const std::string& idx) const;

    std::shared_ptr<Entry> find_child(const std::string&);

    std::shared_ptr<const Entry> find_child(const std::string&) const;
};

template <typename... Args>
void unpack(Args&&... args) {}

template <typename Backend, TypeTag TAG, template <typename, TypeTag> class... Policies>
class EntryImplement : public virtual Entry,
                       public std::enable_shared_from_this<Entry>,
                       public EntryPolicyBody<Backend>,
                       public EntryPolicyAttributes<Backend>,
                       public EntryPolicy<Backend, TAG>,
                       public Policies<Backend, TAG>...
{
private:
    std::shared_ptr<Entry> m_parent_;

public:
    typedef EntryImplement<Backend, TAG, Policies...> this_type;

    EntryImplement() : m_parent_(nullptr), Policies<Backend, TAG>()... {}

    EntryImplement(const std::shared_ptr<Entry>& p) : m_parent_(p), Policies<Backend, TAG>()... {}

    EntryImplement(Entry* p) : EntryImplement(std::shared_ptr<Entry>(p)) {}

    EntryImplement(const this_type& other) : m_parent_(other.m_parent_), Policies<Backend, TAG>(other)... {};

    EntryImplement(this_type&& other) : m_parent_(std::move(other.m_parent_)), Policies<Backend, TAG>(std::forward<this_type>(other))... {};

    virtual ~EntryImplement() = default;

    void swap(this_type& other)
    {
        std::swap(m_parent_, other.m_parent_);
        unpack(Policies<Backend, TAG>::swap(*this)...);
    }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    std::shared_ptr<Entry> self() final { return this->shared_from_this(); }

    std::shared_ptr<Entry> parent() final { return m_parent_; }

    std::shared_ptr<Entry> create() const final { return std::dynamic_pointer_cast<Entry>(std::make_shared<this_type>(m_parent_)); }

    std::shared_ptr<Entry> copy() const final { return std::dynamic_pointer_cast<Entry>(std::make_shared<this_type>(*this)); }
};

std::shared_ptr<Entry>
create_entry(const std::string& backend = "");

} // namespace sp

#endif // SP_ENTRY_H_
