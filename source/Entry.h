#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Node.h"
#include <any>
#include <memory>

namespace sp
{

template <TypeTag>
class EntryInterface;

template <>
struct EntryInterface<TypeTag::Null>
{
    EntryInterface() = default;

    virtual ~EntryInterface() = default;

    virtual EntryInterface<TypeTag::Null>* create() const = 0;

    virtual EntryInterface<TypeTag::Null>* copy() const = 0;

    virtual Node* create_child() = 0;

    virtual void resolve() = 0;

    virtual TypeTag type_tag() const { return TypeTag::Null; }

    bool is_leaf() const { return !(type_tag() == TypeTag::Array || type_tag() == TypeTag::Table); }

    EntryInterface<TypeTag::Null>* as_interface(TypeTag tag);

    template <TypeTag TAG>
    EntryInterface<TAG>* as()
    {
        if (type_tag() == TAG)
        {
            return dynamic_cast<EntryInterface<TAG>*>(this);
        }
        else
        {
            return dynamic_cast<EntryInterface<TAG>*>(as_interface(TAG));
        }
    }

    EntryInterface<TypeTag::Scalar>* as_scalar() { return as<TypeTag::Scalar>(); }

    EntryInterface<TypeTag::Block>* as_block() { return as<TypeTag::Block>(); }

    EntryInterface<TypeTag::Tree>* as_tree() { return as<TypeTag::Tree>(); }

    EntryInterface<TypeTag::Array>* as_array() { return as<TypeTag::Array>(); }

    EntryInterface<TypeTag::Table>* as_table() { return as<TypeTag::Table>(); }

    //-------------------------------------------------------
    // attributes
    virtual bool has_attribute(std::string const& key) const = 0;

    virtual bool check_attribute(std::string const& key, std::any const& v) const = 0;

    virtual void set_attribute(const std::string&, const std::any&) = 0;

    virtual std::any get_attribute(const std::string&) const = 0;

    virtual std::any get_attribute(std::string const& key, std::any const& default_value) = 0;

    virtual void remove_attribute(const std::string&) = 0;

    virtual Range<Iterator<std::pair<std::string, std::any>>> attributes() const = 0;

    virtual void clear_attributes() = 0;
};

template <>
struct EntryInterface<TypeTag::Scalar> : public EntryInterface<TypeTag::Null>
{
    TypeTag type_tag() const final { return TypeTag::Scalar; }

    virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

    virtual void set_scalar(std::any const&) = 0;
};

template <typename Backend>
struct EntryPolicyScalar : public EntryInterface<TypeTag::Scalar>
{

private:
    Backend* m_backend_;

public:
    EntryPolicyScalar(Backend* p) : m_backend_(p){};
    ~EntryPolicyScalar() = default;
    std::any get_scalar() const final;
    void set_scalar(std::any const&) final;
};

template <>
struct EntryInterface<TypeTag::Block> : public EntryInterface<TypeTag::Null>
{
    TypeTag type_tag() const final { return TypeTag::Block; }
    // as leaf node
    virtual std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const = 0; // get block

    virtual void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                               const std::type_info& /*element type*/,
                               const std::vector<size_t>& /*dimensions*/) = 0; // set block
};

template <typename Backend>
struct EntryPolicyBlock : public EntryInterface<TypeTag::Block>
{

private:
    Backend* m_backend_;

public:
    EntryPolicyBlock(Backend* p) : m_backend_(p){};
    ~EntryPolicyBlock() = default;

    // as leaf node
    std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const final; // get block

    void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                       const std::type_info& /*element type*/,
                       const std::vector<size_t>& /*dimensions*/) final; // set block
};

template <>
struct EntryInterface<TypeTag::Tree> : public EntryInterface<TypeTag::Null>
{

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

template <typename Backend>
struct EntryPolicyTree
    : public EntryInterface<TypeTag::Tree>
{
protected:
    Backend* m_backend_;

public:
    EntryPolicyTree(Backend* p) : m_backend_(p){};
    ~EntryPolicyTree() = default;

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

template <>
struct EntryInterface<TypeTag::Array>
    : public EntryInterface<TypeTag::Tree>
{
    TypeTag type_tag() const final { return TypeTag::Array; }

    // as array

    virtual std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr) = 0;

    virtual std::shared_ptr<Node> push_back(Node&&) = 0;

    virtual std::shared_ptr<Node> push_back(const Node&) = 0;

    virtual Node::range push_back(const Node::iterator& b, const Node::iterator& e) = 0;

    virtual std::shared_ptr<Node> at(int idx) = 0;

    virtual std::shared_ptr<const Node> at(int idx) const = 0;

    virtual std::shared_ptr<Node> find_child(size_t) = 0;

    virtual std::shared_ptr<const Node> find_child(size_t) const = 0;
};

template <typename Backend>
struct EntryPolicyArray
    : public EntryInterface<TypeTag::Array>,
      public EntryPolicyTree<Backend>
{
public:
    EntryPolicyArray(Backend* p) : EntryPolicyTree<Backend>(p){};
    ~EntryPolicyArray() = default;

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

template <>
struct EntryInterface<TypeTag::Table>
    : public EntryInterface<TypeTag::Tree>
{ // as table
    TypeTag type_tag() const final { return TypeTag::Table; }

    virtual Node::const_range_kv items() const = 0;

    virtual Node::range_kv items() = 0;

    virtual std::shared_ptr<Node> insert(const std::string& k, std::shared_ptr<Node> const& node) = 0;

    virtual Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e) = 0;

    virtual std::shared_ptr<Node> at(const std::string& key) = 0;

    virtual std::shared_ptr<const Node> at(const std::string& idx) const = 0;

    virtual std::shared_ptr<Node> find_child(const std::string&) = 0;

    virtual std::shared_ptr<const Node> find_child(const std::string&) const = 0;
};

template <typename Backend>
struct EntryPolicyTable
    : public EntryInterface<TypeTag::Table>,
      public EntryPolicyTree<Backend>
{
public:
    EntryPolicyTable(Backend* p) : EntryPolicyTree<Backend>(p){};
    ~EntryPolicyTable() = default;

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

template <typename Backend, template <typename> class... Policies>
class Entry;

template <typename Backend>
class Entry<Backend> : public EntryInterface<TypeTag::Null>
{
protected:
    std::shared_ptr<Backend> m_backend_;

public:
    typedef Entry<Backend> this_type;

    Entry(std::shared_ptr<Backend> p = nullptr) : m_backend_(p != nullptr ? p : std::make_shared<Backend>()){};

    Entry(Backend* p) : Entry(std::shared_ptr<Backend>(p)){};

    Entry(const Entry& other) : Entry(other.m_backend_){};

    Entry(Entry&& other) : Entry(other.m_backend_){};

    virtual ~Entry() = default;

    void swap(this_type& other) { std::swap(m_backend_, other.m_backend_); }

    this_type& operator=(this_type const& other)
    {
        Entry(other).swap(*this);
        return *this;
    }

    EntryInterface<TypeTag::Null>* create() const override { return new this_type(m_backend_->create()); }

    EntryInterface<TypeTag::Null>* copy() const override { return new this_type(*this); }

    Node* create_child() final;

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

template <typename Backend, template <typename> class Policy, template <typename> class... OtherPolicies>
class Entry<Backend, Policy, OtherPolicies...> : public Entry<Backend>,
                                                 public Policy<Backend>,
                                                 public OtherPolicies<Backend>...
{
public:
    typedef Entry<Backend, Policy, OtherPolicies...> this_type;
    typedef Entry<Backend> base_type;

    template <typename... Args>
    Entry(Args&&... args) : base_type(std::forward<Args>(args)...),
                            Policy<Backend>(base_type::m_backend_.get()),
                            OtherPolicies<Backend>(base_type::m_backend_.get())... {};

    ~Entry() = default;

    template <template <typename> class... Others>
    this_type& operator=(Entry<Backend, Others...> const& other)
    {
        base_type::swap(other);
        return *this;
    }

    EntryInterface<TypeTag::Null>* create() const final { return new this_type(base_type::m_backend_->create()); }

    EntryInterface<TypeTag::Null>* copy() const final { return new this_type(*this); }

    TypeTag type_tag() const { return Policy<Backend>::type_tag(); }
};

EntryInterface<TypeTag::Null>* create_entry(const std::string& backend = "");

} // namespace sp

#endif // SP_ENTRY_H_
