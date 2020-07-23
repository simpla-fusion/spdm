#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Node.h"
#include <any>
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

class Entry : public std::enable_shared_from_this<Entry>
{
public:
    Entry() = default;

    Entry(const Entry& other) = default;

    Entry(Entry&& other) = default;

    virtual ~Entry() = default;

    static std::unique_ptr<Entry> create(const std::string& rpath = "");

    static bool add_creator(const std::string& c_id, const std::function<Entry*()>&);

    virtual Node::Type type() const = 0;

    virtual std::shared_ptr<Entry> copy() const = 0;

    virtual std::shared_ptr<Entry> duplicate() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    virtual bool has_attribute(const std::string& name) const = 0;

    virtual Node::element_t get_attribute_raw(const std::string& name) const = 0;

    virtual void set_attribute_raw(const std::string& name, const Node::element_t& value) = 0;

    virtual void remove_attribute(const std::string& name) = 0;

    virtual std::map<std::string, Node::element_t> attributes() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_single(const Node::element_t&) = 0;
    virtual Node::element_t get_single() const = 0;

    virtual void set_tensor(const Node::tensor_t&) = 0;
    virtual Node::tensor_t get_tensor() const = 0;

    virtual void set_block(const Node::block_t&) = 0;
    virtual Node::block_t get_block() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // container
    virtual size_t size() const = 0;

    virtual std::shared_ptr<Entry> parent() const = 0;

    virtual std::shared_ptr<Entry> next() const = 0;

    virtual bool not_equal(const std::shared_ptr<Entry>& other) const = 0;

    virtual bool equal(const std::shared_ptr<Entry>&) const = 0;

    // as array

    virtual std::shared_ptr<Entry> push_back() = 0;

    virtual std::shared_ptr<Entry> pop_back() = 0;

    virtual std::shared_ptr<Entry> item(int idx) const = 0;

    // as object
    virtual std::shared_ptr<Entry> insert(const std::string& path) = 0;
    virtual std::shared_ptr<Entry> insert_r(const std::string& path) = 0;

    virtual std::shared_ptr<Entry> find(const std::string& path) const = 0;
    virtual std::shared_ptr<Entry> find_r(const std::string& path) const = 0;

    virtual std::shared_ptr<Entry> first_child() const = 0;

    virtual void remove(const std::string& path) = 0;
};

template <typename Impl>
class EntryImplement : public Entry
{

public:
    EntryImplement();

    EntryImplement(const Impl&);

    EntryImplement(const EntryImplement&);

    EntryImplement(EntryImplement&&);

    ~EntryImplement();

    void swap(EntryImplement& other);

    Node::Type type() const override;

    std::shared_ptr<Entry> copy() const override;

    std::shared_ptr<Entry> duplicate() const override;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    bool has_attribute(const std::string& name) const override;

    Node::element_t get_attribute_raw(const std::string& name) const override;

    void set_attribute_raw(const std::string& name, const Node::element_t& value) override;

    void remove_attribute(const std::string& name) override;

    std::map<std::string, Node::element_t> attributes() const override;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_single(const Node::element_t&) override;
    Node::element_t get_single() const override;

    void set_tensor(const Node::tensor_t&) override;
    Node::tensor_t get_tensor() const override;

    void set_block(const Node::block_t&) override;
    Node::block_t get_block() const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    std::shared_ptr<Entry> next() const override;

    bool not_equal(const std::shared_ptr<Entry>& other) const override;

    bool equal(const std::shared_ptr<Entry>&) const override;

    // container
    size_t size() const override;

    std::shared_ptr<Entry> parent() const override;

    // as array
    std::shared_ptr<Entry> push_back() override;

    std::shared_ptr<Entry> pop_back() override;

    std::shared_ptr<Entry> item(int idx) const override;

    // as object
    std::shared_ptr<Entry> insert(const std::string& key) override;
    std::shared_ptr<Entry> insert_r(const std::string& path) override;

    std::shared_ptr<Entry> find(const std::string& key) const override;
    std::shared_ptr<Entry> find_r(const std::string& path) const override;

    void remove(const std::string& path) override;

    std::shared_ptr<Entry> first_child() const override;

private:
    Impl m_pimpl_;
    static bool is_registered;
};

#define SP_REGISTER_ENTRY(_NAME_, _CLASS_)            \
    template <>                                       \
    bool sp::EntryImplement<_CLASS_>::is_registered = \
        Entry::add_creator(                           \
            __STRING(_NAME_),                         \
            []() { return dynamic_cast<Entry*>(new EntryImplement<_CLASS_>()); });

} // namespace sp

#endif //SP_ENTRY_H_