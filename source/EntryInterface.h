#ifndef SP_ENTRY_INTERFACE_H_
#define SP_ENTRY_INTERFACE_H_
#include "Entry.h"
#include "Iterator.h"
#include "Range.h"
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

class EntryInterface : public std::enable_shared_from_this<EntryInterface>
{
protected:
    Entry* m_self_;

public:
    EntryInterface();

    EntryInterface(const EntryInterface& other);

    EntryInterface(EntryInterface&& other);

    static std::unique_ptr<EntryInterface> create(const std::string& rpath = "");

    static bool add_creator(const std::string& c_id, const std::function<EntryInterface*()>&);

    virtual ~EntryInterface() = default;

    virtual Entry::Type type() const = 0;

    virtual std::shared_ptr<EntryInterface> copy() const = 0;

    virtual std::shared_ptr<EntryInterface> duplicate() const = 0;

    // virtual int fetch(const std::string& uri) = 0;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    virtual bool has_attribute(const std::string& name) const = 0;

    virtual Entry::single_t get_attribute_raw(const std::string& name) const = 0;

    virtual void set_attribute_raw(const std::string& name, const Entry::single_t& value) = 0;

    virtual void remove_attribute(const std::string& name) = 0;

    virtual std::map<std::string, Entry::single_t> attributes() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_single(const Entry::single_t&) = 0;
    virtual Entry::single_t get_single() const = 0;

    virtual void set_tensor(const Entry::tensor_t&) = 0;
    virtual Entry::tensor_t get_tensor() const = 0;

    virtual void set_block(const Entry::block_t&) = 0;
    virtual Entry::block_t get_block() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // container
    virtual size_t size(const std::string& path = "") const = 0;

    virtual std::shared_ptr<EntryInterface> parent() = 0;

    virtual Range<std::string, std::shared_ptr<EntryInterface>> children(const std::string& path = "") const = 0;

    // as array

    virtual std::shared_ptr<EntryInterface> push_back(const std::string& path) = 0;

    virtual std::shared_ptr<EntryInterface> pop_back(const std::string& path) = 0;

    virtual std::shared_ptr<EntryInterface> item(int idx, const std::string& path = "") = 0;

    virtual Range<std::shared_ptr<EntryInterface>> items(const std::string& path = "") = 0;

    // as object
    virtual std::shared_ptr<EntryInterface> insert(const std::string& path) = 0;

    virtual std::shared_ptr<EntryInterface> find(const std::string& path) = 0;

    virtual std::shared_ptr<EntryInterface> find(const std::string& path) const = 0;

    virtual void remove(const std::string& path) = 0;
};

template <typename Impl>
class EntryImplement : public EntryInterface
{

public:
    using EntryInterface::m_self_;

    EntryImplement();
    EntryImplement(const Impl&);
    EntryImplement(const EntryImplement&);
    EntryImplement(EntryImplement&&);
    ~EntryImplement();

    void swap(EntryImplement& other);

    Entry::Type type() const override;

    std::shared_ptr<EntryInterface> copy() const override;

    std::shared_ptr<EntryInterface> duplicate() const override;

    // int fetch(const std::string& uri) override;
    std::shared_ptr<EntryInterface> parent() const override;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    bool has_attribute(const std::string& name) const override;

    Entry::single_t get_attribute_raw(const std::string& name) const override;

    void set_attribute_raw(const std::string& name, const Entry::single_t& value) override;

    void remove_attribute(const std::string& name) override;

    std::map<std::string, Entry::single_t> attributes() const override;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_single(const Entry::single_t&) override;
    Entry::single_t get_single() const override;

    void set_tensor(const Entry::tensor_t&) override;
    Entry::tensor_t get_tensor() const override;

    void set_block(const Entry::block_t&) override;
    Entry::block_t get_block() const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // container
    size_t size(const std::string& path = "") const;

    std::shared_ptr<EntryInterface> parent();

    Range<std::string, std::shared_ptr<EntryInterface>> children(const std::string& path = "") const;

    // as array

    std::shared_ptr<EntryInterface> push_back(const std::string& path = "") override;

    std::shared_ptr<EntryInterface> pop_back(const std::string& path = "") override;

    std::shared_ptr<EntryInterface> item(int idx, const std::string& path = "") override;

    Range<std::shared_ptr<EntryInterface>> items(const std::string& path = "") override;

    // as object
    std::shared_ptr<EntryInterface> insert(const std::string& path) override;

    std::shared_ptr<EntryInterface> find(const std::string& path) override;

    std::shared_ptr<EntryInterface> find(const std::string& path) const override;

    void remove(const std::string& path) override;

private:
    Impl m_pimpl_;
    static bool is_registered;
};

#define SP_REGISTER_ENTRY(_NAME_, _CLASS_)            \
    template <>                                       \
    bool sp::EntryImplement<_CLASS_>::is_registered = \
        EntryInterface::add_creator(                  \
            __STRING(_NAME_),                         \
            []() { return dynamic_cast<EntryInterface*>(new EntryImplement<_CLASS_>()); });

} // namespace sp

#endif //SP_ENTRY_INTERFACE_H_