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

    EntryObjectPlugin(Entry* self, const XPath&) : EntryObject(self) {}

    EntryObjectPlugin(Entry* self, const Container& container) : EntryObject(self), m_container_(container) {}

    EntryObjectPlugin(Entry* self, Container&& container) : EntryObject(self), m_container_(std::move(container)) {}

    EntryObjectPlugin(const this_type& other) : EntryObject(nullptr), m_container_(other.m_container_) {}

    EntryObjectPlugin(this_type&& other) : EntryObject(nullptr), m_container_(std::move(other.m_container_)) {}

    ~EntryObjectPlugin() = default;

    std::shared_ptr<EntryObject> copy() const override { return std::shared_ptr<EntryObject>(new this_type(*this)); }

    void fetch(const XPath&){};

    void update(const XPath&){};

    //----------------------------------------------------------------------------------------------------------

    size_t size() const override;

    void clear() override;

    //------------------------------------------------------------------

    Cursor<Entry> select(const XPath& path) override;

    Cursor<const Entry> select(const XPath& path) const override;

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> kv_items() override;

    Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> kv_items() const override;

    //------------------------------------------------------------------

    std::shared_ptr<Entry> insert(const std::string& path) override;

    std::shared_ptr<Entry> insert(const XPath& path) override;

    std::shared_ptr<const Entry> get(const std::string& path) const override;

    std::shared_ptr<const Entry> get(const XPath& path) const override;

    void erase(const std::string& path) override;

    void erase(const XPath& path) override;
};

template <typename Container>
class EntryArrayPlugin : public EntryArray
{
private:
    Container m_container_;

public:
    typedef Entry::type_tags type_tags;
    typedef EntryArrayPlugin this_type;

    EntryArrayPlugin(Entry* self) : EntryArray(self) {}

    ~EntryArrayPlugin() = default;

    EntryArrayPlugin(const this_type& other) : EntryArray(nullptr), m_container_(other.m_container_) {}

    EntryArrayPlugin(EntryArrayPlugin&& other) : EntryArray(nullptr), m_container_(std::move(other.m_container_)) {}
    //--------------------------------------------------------------------------------------
    // as array

    std::shared_ptr<EntryArray> copy() const override { return std::dynamic_pointer_cast<EntryArray>(std::make_shared<EntryArrayPlugin>(*this)); }

    size_t size() const override;

    void resize(std::size_t num) override;

    void clear() override;

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    //--------------------------------------------------------------------------------------
    std::shared_ptr<Entry> push_back() override;

    void pop_back() override;

    std::shared_ptr<Entry> get(int idx) override;

    std::shared_ptr<const Entry> get(int idx) const override;
};

#define SPDB_ENTRY_REGISTER(_NAME_, _CLASS_)                                  \
    template <>                                                               \
    bool ::sp::db::EntryObjectPlugin<_CLASS_>::is_registered =                \
        ::sp::utility::Factory<::sp::db::EntryObject, ::sp::db::Entry*>::add( \
            __STRING(_NAME_),                                                 \
            [](::sp::db::Entry* s) { return dynamic_cast<::sp::db::EntryObject*>(new ::sp::db::EntryObjectPlugin<_CLASS_>(s)); });

#define SPDB_ENTRY_ASSOCIATE(_NAME_, _CLASS_, ...)             \
    template <>                                                \
    int ::sp::db::EntryObjectPlugin<_CLASS_>::associated_num = \
        ::sp::utility::Factory<::sp::db::EntryObject, ::sp::db::Entry*>::associate(__STRING(_NAME_), __VA_ARGS__);
} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_