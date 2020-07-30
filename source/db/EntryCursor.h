#ifndef SPDB_ENTRY_CURSOR_H_
#define SPDB_ENTRY_CURSOR_H_
#include "Entry.h"
#include "EntryPlugin.h"
#include "utility/Cursor.h"
#include "utility/Logger.h"
#include <any>
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
namespace sp::db
{

template <>
class EntryPlugin<Cursor<Entry>> : public Entry
{

public:
    typedef EntryCursorProxy<U...> impl_type;

    typedef EntryPlugin<impl_type> this_type;

    EntryPlugin(const impl_type& impl) : m_pimpl_(impl){};

    EntryPlugin(impl_type&& impl) : m_pimpl_(std::move(impl)){};

    template <typename... Args>
    EntryPlugin(Args&&... args) : EntryPlugin(impl_type(std::forward<Args>(args)...)) {}

    EntryPlugin(const this_type& other) : EntryPlugin(other.m_pimpl_){};

    EntryPlugin(this_type&& other) : EntryPlugin(std::move(other.m_pimpl_)){};

    ~EntryPlugin(){};

    std::shared_ptr<Entry> copy() const override { return std::make_shared<this_type>(*this); }

    void swap(EntryPlugin& other) { std::swap(m_pimpl_, other.m_pimpl_); }

    //----------------------------------------------------------------------------------------------------------

    Entry::NodeType type() const override { return self().type(); }

    std::string path() const override { return self().path(); }

    std::string name() const override { return self().name(); }
    //----------------------------------------------------------------------------------------------------------
    // attribute

    bool has_attribute(const std::string& name) const override { return self().has_attribute(name); }

    Entry::element_t get_attribute_raw(const std::string& name) const override { return self().get_attribute_raw(name); }

    void set_attribute_raw(const std::string& name, const element_t& value) override { self().set_attribute_raw(name, value); }

    void remove_attribute(const std::string& name) override { self().remove_attribute(name); }

    std::map<std::string, Entry::element_t> attributes() const override { return self().attributes(); }

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------

    void set_element(const element_t& value) override { self().set_element(value); }

    Entry::element_t get_element() const override { return self().get_element(); }

    void set_tensor(const tensor_t& value) override { self().set_tensor(value); }

    Entry::tensor_t get_tensor() const override { return self().get_tensor(); }

    void set_block(const block_t& value) override { self().set_block(value); }

    Entry::block_t get_block() const override { return self().get_block(); }

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    // as cursor

    size_t size() const override { return m_pimpl_.size(); }

    std::shared_ptr<Entry> next() const override
    {
        auto res = std::make_shared<this_type>(std::move(m_pimpl_.copy()));

        res->m_pimpl_.next();

        return (res->m_pimpl_.get_pointer() != nullptr) ? res : std::shared_ptr<Entry>{nullptr};
    }

    bool same_as(const Entry* other) const override
    {
        return m_pimpl_.get_pointer().get() == other ||
               (other != nullptr && other->same_as(m_pimpl_.get_pointer().get()));
    }; // check

    // container

    std::shared_ptr<Entry> parent() const override { return self().parent(); }

    std::shared_ptr<Entry> first_child() const override { return self().first_child(); }
    // as array

    std::shared_ptr<Entry> push_back() override { return self().push_back(); }

    std::shared_ptr<Entry> pop_back() override { return self().pop_back(); }

    std::shared_ptr<Entry> item(int idx) const override { return self().item(idx); }
    // as object

    std::shared_ptr<Entry> insert(const std::string& key) override { return self().insert(key); }

    std::shared_ptr<Entry> insert_r(const std::string& path) override { return self().insert_r(path); }

    std::shared_ptr<Entry> find(const std::string& key) const override { return self().find(key); }

    std::shared_ptr<Entry> find_r(const std::string& path) const override { return self().find_r(path); }

    void remove(const std::string& path) override { self().remove(path); }

private:
    impl_type m_pimpl_;
    const Entry& self() const
    {
        auto p = m_pimpl_.get_pointer();
        if (p == nullptr)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING + "Empty entry");
        }
        return *p;
    }
    Entry& self()
    {
        auto p = m_pimpl_.get_pointer();
        if (p == nullptr)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING + "Empty entry");
        }
        return *p;
    }
};

template <typename Plugin, typename IT, typename... Args>
std::shared_ptr<Entry> make_iterator(const IT& ib, const IT& ie, Args&&... args)
{
    return std::make_shared<EntryPlugin<EntryCursorProxy<Plugin, IT>>>(ib, ie, std::forward<Args>(args)...);
}
} // namespace sp::db
#endif // SPDB_ENTRY_CURSOR_H_