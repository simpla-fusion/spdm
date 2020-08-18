#ifndef SPDB_ENTRY_PLUGIN_H_
#define SPDB_ENTRY_PLUGIN_H_
#include "Entry.h"
#include "XPath.h"
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

template <typename Container>
class EntryObjectPlugin : public EntryObject
{
private:
    Container m_container_;
    static bool is_registered;
    static int associated_num;

public:
    typedef EntryObjectPlugin<Container> this_type;
    typedef Entry::type_tags type_tags;

    EntryObjectPlugin(Entry* self) : EntryObject(self) {}

    EntryObjectPlugin(const this_type& other) : EntryObject(nullptr), m_container_(other.m_container_) {}

    EntryObjectPlugin(this_type&& other) : EntryObject(nullptr), m_container_(std::move(other.m_container_)) {}

    ~EntryObjectPlugin() = default;

    std::unique_ptr<EntryObject> copy() const override { return std::unique_ptr<EntryObject>(new this_type(*this)); }

    //----------------------------------------------------------------------------------------------------------

    size_t size() const override;

    void clear() override;

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    Cursor<std::pair<const std::string, Entry>> kv_items() override;

    Cursor<const std::pair<const std::string, Entry>> kv_items() const override;

    void insert(const XPath& path, const Entry&) override;

    const Entry fetch(const XPath& path) const override;

    void erase(const XPath& path) override;

    Cursor<Entry> select(const XPath& path) override;

    Cursor<const Entry> select(const XPath& path) const override;
};

#define SPDB_ENTRY_REGISTER(_NAME_, _CLASS_)                                                    \
    template <>                                                                                 \
    bool ::sp::db::EntryObjectPlugin<_CLASS_>::is_registered =                                  \
        ::sp::utility::Factory<::sp::db::EntryObject, : std::shared_ptr<::sp::db::Entry>>::add( \
            __STRING(_NAME_),                                                                   \
            [](std::shared_ptr<::sp::db::Entry> s) { return dynamic_cast<::sp::db::EntryObject*>(new ::sp::db::EntryObjectPlugin<_CLASS_>(s)); });

#define SPDB_ENTRY_ASSOCIATE(_NAME_, _CLASS_, ...)             \
    template <>                                                \
    int ::sp::db::EntryObjectPlugin<_CLASS_>::associated_num = \
        ::sp::utility::Factory < ::sp::db::EntryObject,        \
        std::shared_ptr<::sp::db::Entry>::associate(__STRING(_NAME_), __VA_ARGS__);

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_