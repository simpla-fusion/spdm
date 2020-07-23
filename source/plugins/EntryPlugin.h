#ifndef SP_ENTRY_PLUGIN_H_
#define SP_ENTRY_PLUGIN_H_
#include "../Entry.h"
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
    EntryPlugin();

    EntryPlugin(const Impl&);

    EntryPlugin(const EntryPlugin&);

    EntryPlugin(EntryPlugin&&);

    ~EntryPlugin();

    void swap(EntryPlugin& other);

    Type type() const override;

    std::shared_ptr<Entry> copy() const override;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    bool has_attribute(const std::string& name) const override;

    element_t get_attribute_raw(const std::string& name) const override;

    void set_attribute_raw(const std::string& name, const element_t& value) override;

    void remove_attribute(const std::string& name) override;

    std::map<std::string, element_t> attributes() const override;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_element(const element_t&) override;
    element_t get_element() const override;

    void set_tensor(const tensor_t&) override;
    tensor_t get_tensor() const override;

    void set_block(const block_t&) override;
    block_t get_block() const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // container
    size_t size() const override;

    std::shared_ptr<Entry> parent() const override;

    std::shared_ptr<Entry> first_child() const override;

    std::shared_ptr<Entry> next() const override;

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