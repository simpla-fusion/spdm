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
    friend class Entry;
    typedef EntryObjectPlugin<Container> this_type;

    EntryObjectPlugin() = default;
    virtual ~EntryObjectPlugin() = default;

    EntryObjectPlugin(const Container&);
    EntryObjectPlugin(Container&&);

    EntryObjectPlugin(const this_type&);
    EntryObjectPlugin(this_type&&);

    std::unique_ptr<EntryObject> copy() const override { return std::unique_ptr<EntryObject>(new this_type(*this)); }

    void load(const std::string&) override { NOT_IMPLEMENTED; }

    void save(const std::string&) const override { NOT_IMPLEMENTED; }

    std::pair<std::shared_ptr<EntryObject>, Path> full_path() override { return EntryObject::full_path(); }

    std::pair<std::shared_ptr<const EntryObject>, Path> full_path() const override { return EntryObject::full_path(); }

    size_t size() const override;

    void clear() override;

    //-------------------------------------------------------------------------------------------------------------
    // as container

    Entry at(const Path& path) override { return Entry{entry_value_type{shared_from_this()}, path}; }

    Entry at(const Path& path) const override { return Entry{entry_value_type{const_cast<this_type*>(this)->shared_from_this()}, path}; }

    Cursor<entry_value_type> children() override;

    Cursor<const entry_value_type> children() const override;

    // void for_each(std::function<void(const std::string&, entry_value_type&)> const&) override;

    void for_each(std::function<void(const std::string&, const entry_value_type&)> const&) const override;

    // access children

    // entry_value_type insert(const std::string&, entry_value_type) override;

    // entry_value_type find(const std::string& key) const override;

    // void update(const std::string& key, entry_value_type v) override;

    // void remove(const std::string& path) override;

    //------------------------------------------------------------------------------
    // fundamental operation ：
    /**
     *  Create 
     */
    entry_value_type insert(const Path& path, entry_value_type v) override;
    /**
     * Modify
     */
    void update(const Path& path, entry_value_type v) override;
    /**
     * Retrieve
     */
    entry_value_type find(const Path& path = {}) const override;

    /**
     *  Delete 
     */
    void remove(const Path& path = {}) override;

    //------------------------------------------------------------------------------
    // advanced extension functions
    virtual void merge(const EntryObject& other) override { EntryObject::merge(other); }

    virtual void patch(const EntryObject& other) override { EntryObject::patch(other); }

    virtual void update(const EntryObject& other) override { EntryObject::update(other); }

    virtual bool compare(const entry_value_type& other) const override { return EntryObject::compare(other); }

    virtual entry_value_type diff(const entry_value_type& other) const override { return EntryObject::diff(other); }
};

#define SPDB_ENTRY_REGISTER(_NAME_, _CLASS_)                   \
    template <>                                                \
    bool ::sp::db::EntryObjectPlugin<_CLASS_>::is_registered = \
        ::sp::utility::Factory<::sp::db::EntryObject>::add(    \
            __STRING(_NAME_),                                  \
            []() { return dynamic_cast<::sp::db::EntryObject*>(new ::sp::db::EntryObjectPlugin<_CLASS_>()); });

#define SPDB_ENTRY_ASSOCIATE(_NAME_, _CLASS_, ...)             \
    template <>                                                \
    int ::sp::db::EntryObjectPlugin<_CLASS_>::associated_num = \
        ::sp::utility::Factory<::sp::db::EntryObject>::associate(__STRING(_NAME_), __VA_ARGS__);

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_