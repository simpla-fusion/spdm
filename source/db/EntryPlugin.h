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
class EntryPluginObject : public EntryObject
{
public:
    typedef EntryPluginObject<Impl> this_type;
    typedef Entry::type_tags type_tags;

    static bool add_creator(const std::string& c_id, const std::function<EntryObject*()>&);

    static std::shared_ptr<EntryObject> create(const std::string& request = "");

    template <typename... Args>
    EntryPluginObject(Args&&... args) : m_pimpl_(new Impl{std::forward<Args>(args)...}){};

    EntryPluginObject(const this_type& other) : m_pimpl_(new Impl(*other.m_pimpl_)) {}

    EntryPluginObject(EntryPluginObject&& other) : m_pimpl_(other.m_pimpl_->release()) {}

    ~EntryPluginObject() = default;

    std::shared_ptr<EntryObject> copy() const override { return std::shared_ptr<EntryObject>(new this_type(*this)); }

    //----------------------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    size_t size() const override;

    void clear() override;

    // as object

    std::size_t count(const std::string& name) override;

    Cursor<Entry> insert(const std::string& path) override;

    Cursor<Entry> insert(const Path& path) override;

    Cursor<Entry> find(const std::string& path) override;

    Cursor<Entry> find(const Path& path) override;

    Cursor<const Entry> find(const std::string& path) const override;

    Cursor<const Entry> find(const Path& path) const override;

    void erase(const std::string& path) override {}

    void erase(const Path& path) override {}

    Cursor<Entry> first_child() override;

    Cursor<const Entry> first_child() const override;

    // level 1

    Cursor<Entry> select(const std::string& path) override;

    Cursor<Entry> select(const Path& path) override;

    Cursor<const Entry> select(const std::string& path) const override;

    Cursor<const Entry> select(const Path& path) const override;

private:
    std::unique_ptr<Impl> m_pimpl_;
    static bool is_registered;
};

template <typename Impl>
class EntryPluginArray : public EntryArray
{
public:
    typedef Entry::type_tags type_tags;
    typedef EntryPluginArray<Impl> this_type;

    EntryPluginArray() = default;
    ~EntryPluginArray() = default;

    EntryPluginArray(const EntryPluginArray&) = delete;
    EntryPluginArray(EntryPluginArray&&) = delete;
    // as array

    std::shared_ptr<EntryArray> copy() const override;

    size_t size() const override;

    void resize(std::size_t num) override;

    void clear() override;

    Cursor<Entry> push_back() override;

    void pop_back() override;

    Entry& at(int idx) override;

    const Entry& at(int idx) const override;

private:
    std::unique_ptr<Impl> m_pimpl_;
};

#define SPDB_REGISTER_ENTRY(_NAME_, _CLASS_)                   \
    template <>                                                \
    bool ::sp::db::EntryPluginObject<_CLASS_>::is_registered = \
        ::sp::db::EntryObject::add_creator(__STRING(_NAME_), []() { return dynamic_cast<Entry*>(new EntryPluginObject<_CLASS_>()); });

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_