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
    typedef EntryPlugin<Impl> this_type;

    EntryPlugin(const Impl& impl) : m_pimpl_(impl){};

    EntryPlugin(Impl&& impl) : m_pimpl_(std::move(impl)){};

    template <typename... Args>
    EntryPlugin(Args&&... args) : EntryPlugin(Impl(std::forward<Args>(args)...)) {}

    EntryPlugin(const EntryPlugin& other) : EntryPlugin(other.m_pimpl_){};

    EntryPlugin(EntryPlugin&& other) : EntryPlugin(std::move(other.m_pimpl_)){};

    ~EntryPlugin() = default;

    std::shared_ptr<Entry> copy() const override { return std::make_shared<this_type>(*this); }

    void swap(EntryPlugin& other) { std::swap(m_pimpl_, other.m_pimpl_); }

    //----------------------------------------------------------------------------------------------------------

    Type type() const override;

    //----------------------------------------------------------------------------------------------------------
    // attribute
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

    // as cursor
    size_t size() const override { return 0; }

    std::shared_ptr<Entry> next() const override { return nullptr; }

    bool same_as(const Entry* other) const override { return this == other; }; // check

    // container

    std::shared_ptr<Entry> parent() const override;

    std::shared_ptr<Entry> first_child() const override;

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

template <typename Plugin, typename IT, typename Enable = void>
struct EntryCursorProxy;

template <typename Plugin, typename IT>
struct EntryCursorProxy<Plugin, IT,
                        std::enable_if_t<std::is_same_v<
                            std::pair<const std::string, std::shared_ptr<Entry>>,
                            typename std::iterator_traits<IT>::value_type>>>
{
    typedef EntryCursorProxy<Plugin, IT> this_type;
    typedef IT base_iterator;
    typedef std::shared_ptr<Entry> pointer;

    EntryCursorProxy(const base_iterator& ib, const base_iterator& ie) : m_it_(ib), m_ie_(ie) {}

    EntryCursorProxy(base_iterator&& ib, const base_iterator&& ie) : m_it_(std::move(ib)), m_ie_(std::move(ie)) {}

    ~EntryCursorProxy() = default;

    this_type copy() const { return this_type{m_it_, m_ie_}; }

    pointer get_pointer() const { return m_it_ == m_ie_ ? nullptr : m_it_->second; }

    void next() { ++m_it_; }

    size_t size() const { return std::distance(m_it_, m_ie_); }

    base_iterator m_it_, m_ie_;
};
template <typename Plugin, typename IT>
struct EntryCursorProxy<Plugin, IT,
                        std::enable_if_t<std::is_same_v<
                            std::shared_ptr<Entry>,
                            typename std::iterator_traits<IT>::value_type>>>
{
    typedef EntryCursorProxy<Plugin, IT> this_type;
    typedef IT base_iterator;
    typedef std::shared_ptr<Entry> pointer;

    EntryCursorProxy(const base_iterator& ib, const base_iterator& ie) : m_it_(ib), m_ie_(ie) {}

    EntryCursorProxy(base_iterator&& ib, const base_iterator&& ie) : m_it_(std::move(ib)), m_ie_(std::move(ie)) {}

    ~EntryCursorProxy() = default;

    this_type copy() const { return this_type{m_it_, m_ie_}; }

    pointer get_pointer() const { return m_it_ == m_ie_ ? nullptr : *m_it_; }

    void next() { ++m_it_; }

    size_t size() const { return std::distance(m_it_, m_ie_); }

    base_iterator m_it_, m_ie_;
};
template <typename Plugin, typename IT>
class EntryPlugin<EntryCursorProxy<Plugin, IT>> : public Entry
{

public:
    typedef EntryCursorProxy<Plugin, IT> Impl;

    typedef EntryPlugin<Impl> this_type;

    EntryPlugin(const Impl& impl) : m_pimpl_(impl){};

    EntryPlugin(Impl&& impl) : m_pimpl_(std::move(impl)){};

    template <typename... Args>
    EntryPlugin(Args&&... args) : EntryPlugin(Impl(std::forward<Args>(args)...)) {}

    EntryPlugin(const this_type& other) : EntryPlugin(other.m_pimpl_){};

    EntryPlugin(this_type&& other) : EntryPlugin(std::move(other.m_pimpl_)){};

    ~EntryPlugin(){};

    std::shared_ptr<Entry> copy() const override { return std::make_shared<this_type>(*this); }

    void swap(EntryPlugin& other) { std::swap(m_pimpl_, other.m_pimpl_); }

    //----------------------------------------------------------------------------------------------------------

    Entry::Type type() const override { return m_pimpl_.get_pointer()->type(); }

    //----------------------------------------------------------------------------------------------------------
    // attribute

    bool has_attribute(const std::string& name) const override { return m_pimpl_.get_pointer()->has_attribute(name); }

    Entry::element_t get_attribute_raw(const std::string& name) const override { return m_pimpl_.get_pointer()->get_attribute_raw(name); }

    void set_attribute_raw(const std::string& name, const element_t& value) override { m_pimpl_.get_pointer()->set_attribute_raw(name, value); }

    void remove_attribute(const std::string& name) override { m_pimpl_.get_pointer()->remove_attribute(name); }

    std::map<std::string, Entry::element_t> attributes() const override { return m_pimpl_.get_pointer()->attributes(); }

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------

    void set_element(const element_t& value) override { m_pimpl_.get_pointer()->set_element(value); }

    Entry::element_t get_element() const override { return m_pimpl_.get_pointer()->get_element(); }

    void set_tensor(const tensor_t& value) override { m_pimpl_.get_pointer()->set_tensor(value); }

    Entry::tensor_t get_tensor() const override { return m_pimpl_.get_pointer()->get_tensor(); }

    void set_block(const block_t& value) override { m_pimpl_.get_pointer()->set_block(value); }

    Entry::block_t get_block() const override { return m_pimpl_.get_pointer()->get_block(); }

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // as cursor

    size_t size() const override { return m_pimpl_.size(); }

    std::shared_ptr<Entry> next() const override
    {
        auto res = std::make_shared<EntryPlugin<EntryCursorProxy<Plugin, IT>>>(std::move(m_pimpl_.copy()));
        res->m_pimpl_.next();
        return res;
    }

    bool same_as(const Entry* other) const override
    {
        return m_pimpl_.get_pointer().get() == other ||
               (other != nullptr && other->same_as(m_pimpl_.get_pointer().get()));
    }; // check

    // container

    std::shared_ptr<Entry> parent() const override { return m_pimpl_.get_pointer()->parent(); }

    std::shared_ptr<Entry> first_child() const override { return m_pimpl_.get_pointer()->first_child(); }
    // as array

    std::shared_ptr<Entry> push_back() override { return m_pimpl_.get_pointer()->push_back(); }

    std::shared_ptr<Entry> pop_back() override { return m_pimpl_.get_pointer()->pop_back(); }

    std::shared_ptr<Entry> item(int idx) const override { return m_pimpl_.get_pointer()->item(idx); }
    // as object

    std::shared_ptr<Entry> insert(const std::string& key) override { return m_pimpl_.get_pointer()->insert(key); }

    std::shared_ptr<Entry> insert_r(const std::string& path) override { return m_pimpl_.get_pointer()->insert_r(path); }

    std::shared_ptr<Entry> find(const std::string& key) const override { return m_pimpl_.get_pointer()->find(key); }

    std::shared_ptr<Entry> find_r(const std::string& path) const override { return m_pimpl_.get_pointer()->find_r(path); }

    void remove(const std::string& path) override {  m_pimpl_.get_pointer()->remove(path); }

private:
    Impl m_pimpl_;
};

template <typename Plugin, typename IT, typename... Args>
std::shared_ptr<Entry> make_iterator(const IT& ib, const IT& ie, Args&&... args)
{
    return std::make_shared<EntryPlugin<EntryCursorProxy<Plugin, IT>>>(ib, ie, std::forward<Args>(args)...);
}
} // namespace sp
#endif // SP_ENTRY_PLUGIN_H_