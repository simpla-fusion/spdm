#ifndef SPDB_ENTRY_PLUGIN_H_
#define SPDB_ENTRY_PLUGIN_H_
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
namespace sp::db
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

    std::unique_ptr<Entry> copy() const override
    {
        return std::dynamic_pointer_cast<Entry>(std::make_shared<this_type>(*this));
    };

    //  void init(const Attributes& ) override;

    //----------------------------------------------------------------------------------------------------------

    std::size_t type() const override;

    std::string path() const override;

    std::string name() const override;

    //  std::string name() const override;
    //----------------------------------------------------------------------------------------------------------
    // attribute
    bool has_attribute(const std::string& name) const override;

    type_union get_attribute_raw(const std::string& name) const override;

    void set_attribute_raw(const std::string& name, const type_union& value) override;

    void remove_attribute(const std::string& name) override;

    std::map<std::string, type_union> attributes() const override;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_value(const type_union&) override;

    type_union get_value() const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    //as cursor

    size_t size() const override;

    std::shared_ptr<Entry> next() const override; // traversal

    bool same_as(const Entry*) const override; // check

    void clear() override;

    // as tree node

    std::shared_ptr<Entry> parent() const override;

    std::shared_ptr<Entry> first_child() const override;

    // as array

    void resize(std::size_t num) override;

    std::shared_ptr<Entry> push_back() override;

    std::shared_ptr<Entry> pop_back() override;

    std::shared_ptr<const Entry> at(int idx) const override;

    std::shared_ptr<Entry> at(int idx) override;

    // as object

    std::size_t count(const std::string& name) override;

    std::shared_ptr<Entry> insert(const std::string& path) override;

    std::shared_ptr<Entry> insert(const Path& path) override;

    std::shared_ptr<Entry> find(const std::string& path) const override;

    std::shared_ptr<Entry> find(const Path& path) const override;

    void erase(const std::string& path) override;

    void erase(const Path& path) override;

    // level 1

    std::shared_ptr<Entry> select(const std::string& path) const override { return nullptr; };

    std::shared_ptr<Entry> select(const Path& path) const override { return nullptr; };

private:
    Impl m_pimpl_;
    static bool is_registered;
};

#define SPDB_REGISTER_ENTRY(_NAME_, _CLASS_)       \
    template <>                                    \
    bool sp::EntryPlugin<_CLASS_>::is_registered = \
        Entry::add_creator(                        \
            __STRING(_NAME_),                      \
            []() { return dynamic_cast<Entry*>(new EntryPlugin<_CLASS_>()); });

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_