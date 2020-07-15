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

struct Entry
{
    typedef Iterator<Entry> iterator;
    typedef Iterator<const Entry> const_iterator;
    typedef Range<iterator> range;
    typedef Range<const_iterator> const_range;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<Entry>>> iterator_kv;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<const Entry>>> const_iterator_kv;
    typedef Range<iterator_kv> range_kv;
    typedef Range<const_iterator_kv> const_range_kv;

    Entry() = default;

    virtual ~Entry() = default;

    virtual TypeTag type_tag() const { return TypeTag::Null; }

    virtual Entry* create() const = 0;

    virtual Entry* copy() const = 0;

    // virtual Entry* create_child() = 0;

    virtual void resolve() = 0;

    virtual bool has_attribute(std::string const& key) const = 0;

    virtual bool check_attribute(std::string const& key, std::any const& v) const = 0;

    virtual void set_attribute(const std::string&, const std::any&) = 0;

    virtual std::any get_attribute(const std::string&) const = 0;

    virtual std::any get_attribute(std::string const& key, std::any const& default_value) = 0;

    virtual void remove_attribute(const std::string&) = 0;

    virtual Range<Iterator<std::pair<std::string, std::any>>> attributes() const = 0;

    virtual void clear_attributes() = 0;

    virtual Entry* as_interface(TypeTag tag) = 0;

    template <TypeTag I0>
    Entry* as() { return as_interface(I0); }
};

template <TypeTag TAG>
struct EntryInterface;

template <>
struct EntryInterface<TypeTag::Scalar>
    : public virtual Entry
{
    // as scalar
    TypeTag type_tag() const { return TypeTag::Scalar; }

    virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

    virtual void set_scalar(std::any const&) = 0;
};

template <>
struct EntryInterface<TypeTag::Block>
    : public virtual Entry
{
    // as leaf node
    TypeTag type_tag() const { return TypeTag::Block; }

    virtual std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const = 0; // get block

    virtual void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                               const std::type_info& /*element type*/,
                               const std::vector<size_t>& /*dimensions*/) = 0; // set block
};

struct EntryInterfaceTree
    : public virtual Entry
{
    //--------------------------------------------------------
    // as tree node

    virtual size_t size() const = 0;

    virtual Entry::range children() = 0; // reutrn list of children

    virtual Entry::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(Entry::iterator const&) = 0;

    virtual void remove_children(Entry::range const&) = 0;

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

    virtual std::shared_ptr<Entry> find_child(size_t) = 0;

    virtual std::shared_ptr<const Entry> find_child(size_t) const = 0;
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
};

template <typename Backend, TypeTag TAG>
struct EntryPolicy;

template <typename Backend>
struct EntryPolicyBase
    : public virtual Entry
{
    virtual Backend* backend() = 0;

    virtual const Backend* backend() const = 0;

    void resolve();

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
struct EntryPolicy<Backend, TypeTag::Scalar>
    : public virtual EntryInterface<TypeTag::Scalar>,
      public EntryPolicyBase<Backend>
{
    //--------------------------------------------------------------------
    // as scalar
    std::any get_scalar() const;

    void set_scalar(std::any const&);
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Block>
    : public virtual EntryInterface<TypeTag::Block>,
      public EntryPolicyBase<Backend>
{
    // as block
    std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const; // get block

    void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                       const std::type_info& /*element type*/,
                       const std::vector<size_t>& /*dimensions*/); // set block
};

template <typename Backend>
struct EntryPolicy<Backend, TypeTag::Array>
    : public virtual EntryInterface<TypeTag::Array>,
      public EntryPolicyBase<Backend>
{
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

template <typename Backend, TypeTag TAG = TypeTag::Scalar>
class EntryImplement : public EntryPolicy<Backend, TAG>
{
private:
    std::shared_ptr<Backend> m_backend_;

public:
    typedef EntryImplement<Backend, TAG> this_type;

    EntryImplement(const std::shared_ptr<Backend>& p) : m_backend_(p) {}

    EntryImplement(Backend* p) : EntryImplement(std::shared_ptr<Backend>(p)) {}

    EntryImplement() : EntryImplement(std::make_shared<Backend>()) {}

    EntryImplement(const this_type& other) : EntryImplement(other.m_backend_){};

    EntryImplement(this_type&& other) : EntryImplement(std::move(other.m_backend_)){};

    virtual ~EntryImplement() = default;

    Backend* backend() { return m_backend_.get(); }

    const Backend* backend() const { return m_backend_.get(); }

    void swap(this_type& other) { std::swap(m_backend_, other.m_backend_); }

    Entry* create() const { return new this_type(backend()->create()); }

    Entry* copy() const { return new this_type(*this); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    Entry* as_interface(TypeTag tag)
    {
        Entry* res = nullptr;
        switch (tag)
        {
        case TypeTag::Scalar:
            res = new EntryImplement<Backend, TypeTag::Scalar>(m_backend_);
            break;
        case TypeTag::Block:
            res = new EntryImplement<Backend, TypeTag::Block>(m_backend_);
            break;
        case TypeTag::Array:
            res = new EntryImplement<Backend, TypeTag::Array>(m_backend_);
            break;
        case TypeTag::Table:
            res = new EntryImplement<Backend, TypeTag::Table>(m_backend_);
            break;
        default:
            res = new EntryImplement<Backend>(m_backend_);
            break;
        }
        return res;
    }
};

Entry* create_entry(const std::string& backend = "");

} // namespace sp

#endif // SP_ENTRY_H_
