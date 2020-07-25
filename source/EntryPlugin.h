#ifndef SP_ENTRY_PLUGIN_H_
#define SP_ENTRY_PLUGIN_H_
#include "Entry.h"
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
template <typename Impl>
class EntryPlugin : public Entry
{

public:
    typedef EntryPlugin<Impl> this_type;

    template <typename... Args>
    EntryPlugin(Args&&... args) : m_pimpl_(std::forward<Args>(args)...) {}

    EntryPlugin(const std::string& request) : EntryPlugin(request){};

    EntryPlugin(const EntryPlugin& other) : EntryPlugin(other.m_pimpl_){};

    EntryPlugin(EntryPlugin&& other) : EntryPlugin(std::move(other.m_pimpl_)){};

    ~EntryPlugin() = default;

    std::shared_ptr<Entry> copy() const override { return std::make_shared<this_type>(*this); }

    void swap(EntryPlugin& other) { std::swap(m_pimpl_, other.m_pimpl_); }

    void init(const Attributes&){};

    //----------------------------------------------------------------------------------------------------------

    NodeType type() const override { return NodeType::Null; }

    std::string path() const override { return parent() == nullptr ? name() : parent()->path() + "/" + name(); }

    std::string name() const override { return ""; }

    //----------------------------------------------------------------------------------------------------------
    // attribute
    bool has_attribute(const std::string& name) const override { return false; }

    element_t get_attribute_raw(const std::string& name) const override { return element_t{}; }

    void set_attribute_raw(const std::string& name, const element_t& value) override {}

    void remove_attribute(const std::string& name) override {}

    std::map<std::string, element_t> attributes() const override;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_element(const element_t&) override {}
    element_t get_element() const override { return element_t{}; }

    void set_tensor(const tensor_t&) override {}
    tensor_t get_tensor() const override { return tensor_t{nullptr, typeid(nullptr), {}}; }

    void set_block(const block_t&) override {}
    block_t get_block() const override { return block_t{}; }

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // as cursor
    size_t size() const override { return 0; }

    std::shared_ptr<Entry> next() const override { return nullptr; }

    bool same_as(const Entry* other) const override { return this == other; }; // check

    // container

    std::shared_ptr<Entry> parent() const override { return nullptr; }

    std::shared_ptr<Entry> first_child() const override { return nullptr; }

    // as array
    std::shared_ptr<Entry> push_back() override { return nullptr; }

    std::shared_ptr<Entry> pop_back() override { return nullptr; }

    std::shared_ptr<Entry> item(int idx) const override { return nullptr; }

    // as object
    std::shared_ptr<Entry> insert(const std::string& key) override { return nullptr; }

    std::shared_ptr<Entry> insert_r(const std::string& path) override { return nullptr; }

    std::shared_ptr<Entry> find(const std::string& key) const override { return nullptr; }

    std::shared_ptr<Entry> find_r(const std::string& path) const override { return nullptr; }

    void remove(const std::string& path) override {}

private:
    Impl m_pimpl_;
    static bool is_registered;
};

#define SP_REGISTER_ENTRY(_NAME_, _CLASS_)         \
    template <>                                    \
    bool sp::EntryPlugin<_CLASS_>::is_registered = \
        Entry::add_creator(                        \
            __STRING(_NAME_),                      \
            []() { return dynamic_cast<Entry*>(new EntryPlugin<_CLASS_>()); });

} // namespace sp
#endif // SP_ENTRY_PLUGIN_H_