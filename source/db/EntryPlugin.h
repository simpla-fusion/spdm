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
    using typename Entry::type_tags;

    using typename Entry::element;

    using typename Entry::cursor;

    using typename Entry::const_cursor;

    EntryPlugin() = default;

    EntryPlugin(const EntryPlugin& other) = default;

    EntryPlugin(EntryPlugin&& other) = default;

    ~EntryPlugin() = default;

    static std::shared_ptr<Entry> create(const std::string& request = "");

    static bool add_creator(const std::string& c_id, const std::function<Entry*()>&);

    std::shared_ptr<Entry> copy() const override { return std::shared_ptr<Entry>(new EntryPlugin(*this)); }

    //----------------------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_value(const std::string& path, const element&) override;

    element get_value(const std::string& path) const override;

    std::size_t type(const std::string& path) const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    size_t size() const override;

    void clear() override;

    // as object

    std::size_t count(const std::string& name) override;

    cursor insert(const std::string& path) override;

    cursor insert(const Path& path) override;

    cursor find(const std::string& path) override;

    cursor find(const Path& path) override;

    const_cursor find(const std::string& path) const override;

    const_cursor find(const Path& path) const override;

    void erase(const std::string& path) override {}

    void erase(const Path& path) override {}

    cursor first_child() override;

    const_cursor first_child() const override;

    // level 1

    cursor select(const std::string& path) override;

    cursor select(const Path& path) override;

    const_cursor select(const std::string& path) const override;

    const_cursor select(const Path& path) const override;

private:
    std::unique_ptr<Impl> m_pimpl_;
};

template <typename Impl>
class EntryPluginArray : EntryArray
{
public:
    typedef Cursor<typename Entry::element> cursor;

    typedef Cursor<const typename Entry::element> const_cursor;

    EntryPluginArray();

    ~EntryPluginArray() = default;

    // as array

    std::shared_ptr<EntryArray> copy() const override;

    size_t size() const override;

    void resize(std::size_t num) override;

    void clear() override;

    Entry::cursor push_back() override;

    void pop_back() override;

    Entry::element& at(int idx) override;

    const Entry::element& at(int idx) const override;

private:
    std::unique_ptr<Impl> m_pimpl_;
};

#define SPDB_REGISTER_ENTRY(_NAME_, _CLASS_)       \
    template <>                                    \
    bool sp::EntryPlugin<_CLASS_>::is_registered = \
        Entry::add_creator(                        \
            __STRING(_NAME_),                      \
            []() { return dynamic_cast<Entry*>(new EntryPlugin<_CLASS_>()); });

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_