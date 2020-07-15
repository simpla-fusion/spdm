#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Node.h"
#include <any>
#include <memory>

namespace sp
{

template <TypeTag TAG>
struct EntryInterface;

struct EntryInterfaceBase
{
    virtual TypeTag type_tag() const { return TypeTag::Null; }

    virtual EntryInterfaceBase* create() const = 0;

    virtual EntryInterfaceBase* copy() const = 0;

    // virtual EntryInterfaceBase* create_child() = 0;

    virtual void resolve() = 0;

    virtual bool has_attribute(std::string const& key) const = 0;

    virtual bool check_attribute(std::string const& key, std::any const& v) const = 0;

    virtual void set_attribute(const std::string&, const std::any&) = 0;

    virtual std::any get_attribute(const std::string&) const = 0;

    virtual std::any get_attribute(std::string const& key, std::any const& default_value) = 0;

    virtual void remove_attribute(const std::string&) = 0;

    virtual Range<Iterator<std::pair<std::string, std::any>>> attributes() const = 0;

    virtual void clear_attributes() = 0;

    virtual EntryInterfaceBase* as_interface(TypeTag tag) = 0;

    template <TypeTag I0>
    EntryInterfaceBase* as() { return as_interface(I0); }
};

template <>
struct EntryInterface<TypeTag::Scalar>
    : public EntryInterfaceBase
{
    // as scalar
    TypeTag type_tag() const override { return TypeTag::Scalar; }

    virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

    virtual void set_scalar(std::any const&) = 0;
};

template <>
struct EntryInterface<TypeTag::Block>
    : public EntryInterfaceBase
{
    // as leaf node
    TypeTag type_tag() const override { return TypeTag::Block; }

    virtual std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const = 0; // get block

    virtual void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                               const std::type_info& /*element type*/,
                               const std::vector<size_t>& /*dimensions*/) = 0; // set block
};

struct EntryInterfaceTree
    : public EntryInterfaceBase
{
    //--------------------------------------------------------
    // as tree node

    virtual size_t size() const = 0;

    virtual Node::range children() = 0; // reutrn list of children

    virtual Node::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(Node::iterator const&) = 0;

    virtual void remove_children(Node::range const&) = 0;

    virtual Node::iterator begin() = 0;

    virtual Node::iterator end() = 0;

    virtual Node::const_iterator cbegin() const = 0;

    virtual Node::const_iterator cend() const = 0;
};

template <>
struct EntryInterface<TypeTag::Array>
    : public EntryInterfaceTree
{
    // as array
    TypeTag type_tag() const override { return TypeTag::Array; }

    virtual std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr) = 0;

    virtual std::shared_ptr<Node> push_back(Node&&) = 0;

    virtual std::shared_ptr<Node> push_back(const Node&) = 0;

    virtual Node::range push_back(const Node::iterator& b, const Node::iterator& e) = 0;

    virtual std::shared_ptr<Node> at(int idx) = 0;

    virtual std::shared_ptr<const Node> at(int idx) const = 0;

    virtual std::shared_ptr<Node> find_child(size_t) = 0;

    virtual std::shared_ptr<const Node> find_child(size_t) const = 0;
};

template <>
struct EntryInterface<TypeTag::Table>
    : public EntryInterfaceTree
{
    // as table
    TypeTag type_tag() const override { return TypeTag::Table; }

    virtual Node::const_range_kv items() const = 0;

    virtual Node::range_kv items() = 0;

    virtual std::shared_ptr<Node> insert(const std::string& k, std::shared_ptr<Node> const& node) = 0;

    virtual Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e) = 0;

    virtual std::shared_ptr<Node> at(const std::string& key) = 0;

    virtual std::shared_ptr<const Node> at(const std::string& idx) const = 0;

    virtual std::shared_ptr<Node> find_child(const std::string&) = 0;

    virtual std::shared_ptr<const Node> find_child(const std::string&) const = 0;
};

template <typename Backend, TypeTag TAG, typename Interface = EntryInterface<TAG>>
struct EntryPolicy;

template <typename Backend, typename Interface>
struct EntryPolicyBase : public Interface
{
    virtual Backend* backend() = 0;

    virtual const Backend* backend() const = 0;

    void resolve() final;

    //----------------------------------------------------------------
    // attributes

    bool has_attribute(std::string const& key) const final;

    bool check_attribute(std::string const& key, std::any const& v) const final;

    void set_attribute(const std::string&, const std::any&) final;

    std::any get_attribute(const std::string&) const final;

    std::any get_attribute(std::string const& key, std::any const& default_value) final;

    void remove_attribute(const std::string&) final;

    Range<Iterator<std::pair<std::string, std::any>>> attributes() const final;

    void clear_attributes() final;
};

template <typename Backend, typename Interface>
struct EntryPolicy<Backend, TypeTag::Scalar, Interface> : public EntryPolicyBase<Backend, Interface>
{
    //--------------------------------------------------------------------
    // as scalar
    std::any get_scalar() const final;

    void set_scalar(std::any const&) final;
};

template <typename Backend, typename Interface>
struct EntryPolicy<Backend, TypeTag::Block, Interface> : public EntryPolicyBase<Backend, Interface>
{
    // as block
    std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const final; // get block

    void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                       const std::type_info& /*element type*/,
                       const std::vector<size_t>& /*dimensions*/) final; // set block
};

template <typename Backend, typename Interface = EntryInterfaceTree>
struct EntryPolicyTree : public EntryPolicyBase<Backend, Interface>
{
    // as tree node
    size_t size() const final;

    Node::range children() final;

    Node::const_range children() const final;

    void clear_children() final;

    void remove_child(Node::iterator const&) final;

    void remove_children(Node::range const&) final;

    Node::iterator begin() final;

    Node::iterator end() final;

    Node::const_iterator cbegin() const final;

    Node::const_iterator cend() const final;
};

template <typename Backend, typename Interface>
struct EntryPolicy<Backend, TypeTag::Array, Interface> : public EntryPolicyTree<Backend, Interface>
{

    // as array
    std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr) final;

    std::shared_ptr<Node> push_back(Node&&) final;

    std::shared_ptr<Node> push_back(const Node&) final;

    Node::range push_back(const Node::iterator& b, const Node::iterator& e) final;

    std::shared_ptr<Node> at(int idx) final;

    std::shared_ptr<const Node> at(int idx) const final;

    std::shared_ptr<Node> find_child(size_t) final;

    std::shared_ptr<const Node> find_child(size_t) const final;
};

template <typename Backend, typename Interface>
struct EntryPolicy<Backend, TypeTag::Table, Interface> : public EntryPolicyTree<Backend, Interface>
{
    // as table
    Node::const_range_kv items() const final;

    Node::range_kv items() final;

    std::shared_ptr<Node> insert(const std::string& k, std::shared_ptr<Node> const& node) final;

    Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e) final;

    std::shared_ptr<Node> at(const std::string& key) final;

    std::shared_ptr<const Node> at(const std::string& idx) const final;

    std::shared_ptr<Node> find_child(const std::string&) final;

    std::shared_ptr<const Node> find_child(const std::string&) const final;
};

template <typename Backend, TypeTag TAG = TypeTag::Scalar>
class Entry : public EntryPolicy<Backend, TAG>
{
private:
    std::shared_ptr<Backend> m_backend_;

public:
    typedef Entry<Backend, TAG> this_type;

    Entry(const std::shared_ptr<Backend>& p) : m_backend_(p) {}

    Entry(Backend* p) : Entry(std::shared_ptr<Backend>(p)) {}

    Entry() : Entry(std::make_shared<Backend>()) {}

    Entry(const Entry& other) : Entry(other.m_backend_){};

    Entry(Entry&& other) : Entry(std::move(other.m_backend_)){};

    virtual ~Entry() = default;

    Backend* backend() override { return m_backend_.get(); }

    const Backend* backend() const override { return m_backend_.get(); }

    void swap(this_type& other) { std::swap(m_backend_, other.m_backend_); }

    EntryInterfaceBase* create() const override { return new this_type(backend()->create()); }

    EntryInterfaceBase* copy() const override { return new this_type(*this); }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    EntryInterfaceBase* as_interface(TypeTag tag)
    {
        EntryInterfaceBase* res = nullptr;
        switch (tag)
        {
        case TypeTag::Scalar:
            res = new Entry<Backend, TypeTag::Scalar>(m_backend_);
            break;
        case TypeTag::Block:
            res = new Entry<Backend, TypeTag::Block>(m_backend_);
            break;
        case TypeTag::Array:
            res = new Entry<Backend, TypeTag::Array>(m_backend_);
            break;
        case TypeTag::Table:
            res = new Entry<Backend, TypeTag::Table>(m_backend_);
            break;
        default:
            res = new Entry<Backend>(m_backend_);
            break;
        }
        return res;
    }
};

EntryInterfaceBase* create_entry(const std::string& backend = "");

} // namespace sp

#endif // SP_ENTRY_H_
