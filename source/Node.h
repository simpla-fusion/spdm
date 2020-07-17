#ifndef SP_NODE_H_
#define SP_NODE_H_

#include "Range.h"
#include <algorithm>
#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <utility>
#include <variant>
#include <vector>
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
typedef std::variant<bool, int, double, std::string> scalar_t;

/**
 * 
 * 
 *  @ref  Virtual inheritance ,https://en.wikipedia.org/wiki/Virtual_inheritance
*/

template <TypeTag TAG>
struct NodeInterface;

struct NodeInterfaceAttributes
{
    virtual bool has_attribute(std::string const& key) const = 0;

    virtual bool check_attribute(std::string const& key, scalar_t const& v) const = 0;

    virtual void set_attribute(const std::string&, const scalar_t&) = 0;

    virtual scalar_t get_attribute(const std::string&) const = 0;

    virtual scalar_t get_attribute(std::string const& key, scalar_t const& default_value) = 0;

    virtual void remove_attribute(const std::string&) = 0;

    virtual Range<Iterator<std::pair<std::string, scalar_t>>> attributes() const = 0;

    virtual void clear_attributes() = 0;
};
struct Node;

struct iteraotr;

struct NodeInterfaceBase
{

    NodeInterfaceBase() = default;

    virtual ~NodeInterfaceBase() = default;

    virtual TypeTag type_tag() const { return TypeTag::Null; }

    virtual bool is_leaf() const = 0;
    virtual bool is_root() const = 0;
    virtual bool is_emptry() const = 0;

    virtual iterator parent() const = 0;
};

struct Node
    : public virtual NodeInterfaceBase
{
    typedef Iterator<Node> iterator;
    typedef Iterator<const Node> const_iterator;
    typedef Range<iterator> range;
    typedef Range<const_iterator> const_range;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<Node>>> iterator_kv;
    typedef Iterator<std::pair<const std::string, std::shared_ptr<const Node>>> const_iterator_kv;
    typedef Range<iterator_kv> range_kv;
    typedef Range<const_iterator_kv> const_range_kv;
};

template <>
struct NodeInterface<TypeTag::Null>
    : public virtual NodeInterfaceBase
{
    TypeTag type_tag() const { return TypeTag::Null; }
};

template <>
struct NodeInterface<TypeTag::Scalar>
    : public virtual NodeInterfaceBase
{
    // as scalar
    TypeTag type_tag() const { return TypeTag::Scalar; }

    virtual void set_scalar(const scalar_t&) = 0;
    virtual scalar_t get_scalar() const = 0;

    template <typename U>
    void set_value(const U& u) { return set_scalar(scalar_t(u)); }

    template <typename U>
    U get_value() const { return std::get<U>(get_scalar()); }
};

template <>
struct NodeInterface<TypeTag::Block>
    : public virtual NodeInterfaceBase
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

struct NodeInterfaceTree
    : public virtual NodeInterfaceBase
{
    //--------------------------------------------------------
    // as tree node

    virtual size_t size() const = 0;

    virtual Node::range children() = 0; // reutrn list of children

    virtual Node::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(const Node::iterator&) = 0;

    virtual void remove_children(const Node::range&) = 0;

    virtual Node::iterator begin() = 0;

    virtual Node::iterator end() = 0;

    virtual Node::const_iterator cbegin() const = 0;

    virtual Node::const_iterator cend() const = 0;
};

template <>
struct NodeInterface<TypeTag::Array>
    : public virtual NodeInterfaceTree
{
    // as array
    TypeTag type_tag() const { return TypeTag::Array; }

    virtual std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr) = 0;

    virtual std::shared_ptr<Node> push_back(Node&&) = 0;

    virtual std::shared_ptr<Node> push_back(const Node&) = 0;

    virtual Node::range push_back(const Node::iterator& b, const Node::iterator& e) = 0;

    virtual std::shared_ptr<Node> at(int idx) = 0;

    virtual std::shared_ptr<const Node> at(int idx) const = 0;

    Node& operator[](size_t idx) { return *at(idx); }

    const Node& operator[](size_t idx) const { return *at(idx); }
};

template <>
struct NodeInterface<TypeTag::Table>
    : public virtual NodeInterfaceTree
{
    // as table
    TypeTag type_tag() const { return TypeTag::Table; }

    virtual Node::const_range_kv items() const = 0;

    virtual Node::range_kv items() = 0;

    virtual std::shared_ptr<Node> insert(const std::string& k, std::shared_ptr<Node> const& node) = 0;

    virtual Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e) = 0;

    virtual std::shared_ptr<Node> at(const std::string& key) = 0;

    virtual std::shared_ptr<const Node> at(const std::string& idx) const = 0;

    virtual std::shared_ptr<Node> find_child(const std::string&) = 0;

    virtual std::shared_ptr<const Node> find_child(const std::string&) const = 0;

    template <typename TI0, typename TI1>
    auto insert(TI0 const& b, TI1 const& e) { return insert(iterator_kv(b), iterator_kv(e)); }

    Node& operator[](const std::string& path) { return *at(path); }

    const Node& operator[](const std::string& path) const { return *at(path); }
};

template <typename TEntry, TypeTag TAG>
struct NodePolicy;

template <typename TEntry>
struct NodePolicyBase
{
    virtual std::shared_ptr<Node> self() = 0;
    virtual std::shared_ptr<Node> parent() const = 0;
    virtual std::shared_ptr<TEntry> entry() = 0;
    virtual std::shared_ptr<TEntry> entry() const = 0;
    virtual void swap(NodePolicyBase<TEntry>&){};
};

template <typename TEntry>
struct NodePolicyBody
    : public virtual NodeInterfaceBase,
      public virtual NodePolicyBase<TEntry>
{
    void resolve() final;
};

template <typename TEntry>
struct NodePolicy<TEntry, TypeTag::Null>
    : public virtual NodeInterface<TypeTag::Null>,
      public virtual NodePolicyBase<TEntry>
{
    std::shared_ptr<Node> as_interface(TypeTag tag) { return convert_to(tag); }
};

template <typename TEntry>
struct NodePolicy<TEntry, TypeTag::Scalar>
    : public virtual NodeInterface<TypeTag::Scalar>,
      public virtual NodePolicyBase<TEntry>
{
    std::shared_ptr<Node> as_interface(TypeTag tag);

    //--------------------------------------------------------------------
    // as scalar
    void set_scalar(const scalar_t&) final;
    scalar_t get_scalar() const final;
};

template <typename TEntry>
struct NodePolicy<TEntry, TypeTag::Block>
    : public virtual NodeInterface<TypeTag::Block>,
      public virtual NodePolicyBase<TEntry>
{
    std::shared_ptr<Node> as_interface(TypeTag tag);

    // as block
    std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const; // get block

    void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                       const std::type_info& /*element type*/,
                       const std::vector<size_t>& /*dimensions*/); // set block
};

template <typename TEntry>
struct NodePolicy<TEntry, TypeTag::Array>
    : public virtual NodeInterface<TypeTag::Array>,
      public virtual NodePolicyBase<TEntry>
{
    std::shared_ptr<Node> as_interface(TypeTag tag);

    size_t size() const;

    Node::range children();

    Node::const_range children() const;

    void clear_children();

    void remove_child(Node::iterator const&);

    void remove_children(Node::range const&);

    Node::iterator begin();

    Node::iterator end();

    Node::const_iterator cbegin() const;

    Node::const_iterator cend() const;

    // as array
    std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr);

    std::shared_ptr<Node> push_back(Node&&);

    std::shared_ptr<Node> push_back(const Node&);

    Node::range push_back(const Node::iterator& b, const Node::iterator& e);

    std::shared_ptr<Node> at(int idx);

    std::shared_ptr<const Node> at(int idx) const;

    std::shared_ptr<Node> find_child(size_t);

    std::shared_ptr<const Node> find_child(size_t) const;
};

template <typename TEntry>
struct NodePolicy<TEntry, TypeTag::Table>
    : public virtual NodeInterface<TypeTag::Table>,
      public NodePolicyBase<TEntry>
{
    std::shared_ptr<Node> as_interface(TypeTag tag);

    size_t size() const;

    Node::range children();

    Node::const_range children() const;

    void clear_children();

    void remove_child(Node::iterator const&);

    void remove_children(Node::range const&);

    Node::iterator begin();

    Node::iterator end();

    Node::const_iterator cbegin() const;

    Node::const_iterator cend() const;

    // as table
    Node::const_range_kv items() const;

    Node::range_kv items();

    std::shared_ptr<Node> insert(const std::string& k, std::shared_ptr<Node> const& node);

    Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e);

    std::shared_ptr<Node> at(const std::string& key);

    std::shared_ptr<const Node> at(const std::string& idx) const;

    std::shared_ptr<Node> find_child(const std::string&);

    std::shared_ptr<const Node> find_child(const std::string&) const;
};

template <typename... Args>
void unpack(Args&&... args) {}

template <typename TEntry, TypeTag TAG, template <typename, TypeTag> class... Policies>
class NodeImplement : public virtual Node,
                      public std::enable_shared_from_this<Node>,
                      public NodePolicyBody<TEntry>,
                      public NodePolicyAttributes<TEntry>,
                      public NodePolicy<TEntry, TAG>,
                      public Policies<TEntry, TAG>...
{
private:
    std::shared_ptr<Node> m_parent_;
    std::shared_ptr<TEntry> m_entry_;

public:
    typedef NodeImplement<TEntry, TAG, Policies...> this_type;

    NodeImplement(const std::shared_ptr<Node>& p, std::shared_ptr<TEntry> const& b)
        : m_parent_(p),
          m_entry_(b),
          NodePolicyBody<TEntry>(),
          NodePolicyAttributes<TEntry>(),
          NodePolicy<TEntry, TAG>(),
          Policies<TEntry, TAG>()... {}

    NodeImplement() : NodeImplement(nullptr, std::make_shared<TEntry>()) {}

    // template <typename... Args>
    // NodeImplement(const std::shared_ptr<Node>& p, Args&&... args)
    //     : NodeImplement(p, std::make_shared<TEntry>(std::forward<Args>(args)...)) {}

    NodeImplement(const this_type& other)
        : m_parent_(other.m_parent_),
          m_entry_(other.m_entry_),
          NodePolicyBody<TEntry>(other),
          NodePolicyAttributes<TEntry>(other),
          NodePolicy<TEntry, TAG>(other),
          Policies<TEntry, TAG>(other)... {}

    NodeImplement(this_type&& other)
        : m_parent_(std::move(other.m_parent_)),
          m_entry_(std::move(other.m_entry_)),
          NodePolicyBody<TEntry>(std::forward<this_type>(other)),
          NodePolicyAttributes<TEntry>(std::forward<this_type>(other)),
          NodePolicy<TEntry, TAG>(std::forward<this_type>(other)),
          Policies<TEntry, TAG>(std::forward<this_type>(other))... {};

    virtual ~NodeImplement() = default;

    void swap(this_type& other)
    {
        std::swap(m_parent_, other.m_parent_);
        m_entry_.swap(other.m_entry_);
        NodePolicyBody<TEntry>::swap(other);
        NodePolicyAttributes<TEntry>::swap(other);
        NodePolicy<TEntry, TAG>::swap(other);
        unpack(Policies<TEntry, TAG>::swap(other)...);
    }

    this_type& operator=(this_type const& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    std::shared_ptr<Node> self() final { return this->shared_from_this(); }

    std::shared_ptr<Node> parent() const final { return m_parent_; }

    std::shared_ptr<TEntry> entry() { return m_entry_; }

    std::shared_ptr<TEntry> entry() const { return m_entry_; }

    std::shared_ptr<Node> convert_to(TypeTag tag) const final
    {
        std::shared_ptr<Node> res;
        switch (tag)
        {
        case TypeTag::Scalar:
            res.reset(new NodeImplement<TEntry, TypeTag::Scalar, Policies...>(parent(), entry()));
            break;
        case TypeTag::Block:
            res.reset(new NodeImplement<TEntry, TypeTag::Block, Policies...>(parent(), entry()));
            break;
        case TypeTag::Array:
            res.reset(new NodeImplement<TEntry, TypeTag::Array, Policies...>(parent(), entry()));
            break;
        case TypeTag::Table:
            res.reset(new NodeImplement<TEntry, TypeTag::Table, Policies...>(parent(), entry()));
            break;
        default:
            res.reset(new NodeImplement<TEntry, TypeTag::Table, Policies...>(parent(), entry()));
            break;
        }
        return res;
    }

    std::shared_ptr<Node> copy() const final { return std::dynamic_pointer_cast<Node>(std::make_shared<this_type>(*this)); }
};

std::shared_ptr<Node>
create_node(const std::string& backend = "");

} // namespace sp

#endif // SP_NODE_H_