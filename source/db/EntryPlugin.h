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
    typedef traits::pre_tagged_types element_type;

    EntryPlugin() = default;

    EntryPlugin(const EntryPlugin& other) = default;

    EntryPlugin(EntryPlugin&& other) = default;

    ~EntryPlugin() = default;

    static std::unique_ptr<Entry> create(const std::string& request = "");

    static bool add_creator(const std::string& c_id, const std::function<Entry*()>&);

    std::unique_ptr<Entry> copy() const override;

    //----------------------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_value(const std::string& path, const element_type&) override;

    element_type get_value(const std::string& path) const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    size_t size() const override;

    void clear() override;

    // as object

    std::size_t count(const std::string& name) override;

    std::unique_ptr<EntryCursor> insert(const std::string& path) override;

    std::unique_ptr<EntryCursor> insert(const Path& path) override;

    std::unique_ptr<EntryCursor> find(const std::string& path) override;

    std::unique_ptr<EntryCursor> find(const Path& path) override;

    std::unique_ptr<const EntryCursor> find(const std::string& path) const override;

    std::unique_ptr<const EntryCursor> find(const Path& path) const override;

    void erase(const std::string& path) {}

    void erase(const Path& path) {}

    std::unique_ptr<EntryCursor> first_child() override;

    std::unique_ptr<const EntryCursor> first_child() const override;

    // level 1

    std::unique_ptr<EntryCursor> select(const std::string& path) override;

    std::unique_ptr<EntryCursor> select(const Path& path) override;

    std::unique_ptr<const EntryCursor> select(const std::string& path) const override;

    std::unique_ptr<const EntryCursor> select(const Path& path) const override;

private:
    std::unique_ptr<Impl> m_pimpl_;
};

template <typename Impl>
class EntryArrayPlugin : EntryArray
{
public:
    ~EntryArrayPlugin() = default;

    std::unique_ptr<EntryArray> copy() const override;
    ;

    // as array
    size_t size() const { return 0; }

    void resize(std::size_t num){};

    void clear(){};

    std::unique_ptr<EntryCursor> push_back() override;
    ;

    std::unique_ptr<EntryCursor> pop_back() override;
    ;

    std::shared_ptr<const Entry> at(int idx) const override;
    ;

    std::shared_ptr<Entry> at(int idx) override;
    ;

private:
    std::unique_ptr<EntryArray> m_pimpl_;
};

#define SPDB_REGISTER_ENTRY(_NAME_, _CLASS_)       \
    template <>                                    \
    bool sp::EntryPlugin<_CLASS_>::is_registered = \
        Entry::add_creator(                        \
            __STRING(_NAME_),                      \
            []() { return dynamic_cast<Entry*>(new EntryPlugin<_CLASS_>()); });

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_