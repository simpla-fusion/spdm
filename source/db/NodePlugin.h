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
class NodePlugin : public NodeObject
{
private:
    Container m_container_;
    static bool is_registered;
    static int associated_num;

public:
    friend class Entry;
    typedef NodePlugin<Container> this_type;

    NodePlugin() = default;
    virtual ~NodePlugin() = default;

    NodePlugin(const Container&);
    NodePlugin(Container&&);

    NodePlugin(const this_type&);
    NodePlugin(this_type&&);

    std::unique_ptr<NodeObject> copy() const override { return std::unique_ptr<NodeObject>(new this_type(*this)); }

    void load(const std::string&) override { NOT_IMPLEMENTED; }

    void save(const std::string&) const override { NOT_IMPLEMENTED; }

    std::pair<std::shared_ptr<NodeObject>, Path> full_path() override { return NodeObject::full_path(); }

    std::pair<std::shared_ptr<const NodeObject>, Path> full_path() const override { return NodeObject::full_path(); }

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
    virtual void merge(const NodeObject& other) override { NodeObject::merge(other); }

    virtual void patch(const NodeObject& other) override { NodeObject::patch(other); }

    virtual void update(const NodeObject& other) override { NodeObject::update(other); }

    virtual bool compare(const entry_value_type& other) const override { return NodeObject::compare(other); }

    virtual entry_value_type diff(const entry_value_type& other) const override { return NodeObject::diff(other); }
};

#define SPDB_ENTRY_REGISTER(_NAME_, _CLASS_)                   \
    template <>                                                \
    bool ::sp::db::NodePlugin<_CLASS_>::is_registered = \
        ::sp::utility::Factory<::sp::db::NodeObject>::add(    \
            __STRING(_NAME_),                                  \
            []() { return dynamic_cast<::sp::db::NodeObject*>(new ::sp::db::NodePlugin<_CLASS_>()); });

#define SPDB_ENTRY_ASSOCIATE(_NAME_, _CLASS_, ...)             \
    template <>                                                \
    int ::sp::db::NodePlugin<_CLASS_>::associated_num = \
        ::sp::utility::Factory<::sp::db::NodeObject>::associate(__STRING(_NAME_), __VA_ARGS__);

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_