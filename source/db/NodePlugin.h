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
    typedef NodePlugin<Container> this_type;

    NodePlugin() = default;

    virtual ~NodePlugin() = default;

    NodePlugin(const Node& opt) { load(opt); };

    NodePlugin(const std::initializer_list<Node>& opt) { NOT_IMPLEMENTED; }

    NodePlugin(const Container& container) : m_container_(container) {}

    NodePlugin(Container&& container) : m_container_(std::move(container)) {}

    NodePlugin(const this_type& other) : m_container_(other.m_container_) {}

    NodePlugin(this_type&& other) : m_container_(std::move(other.m_container_)) {}

    std::shared_ptr<NodeObject> copy() const override { return std::shared_ptr<NodeObject>(new this_type(*this)); }

    void load(const Node&) override {}

    void save(const Node&) const override { NOT_IMPLEMENTED; }

    Container& container() { return m_container_; }

    const Container& container() const { return m_container_; }

    bool is_same(const NodeObject&) const override
    {
        NOT_IMPLEMENTED;
        return false;
    }

    bool empty() const override
    {
        NOT_IMPLEMENTED;
        return false;
    }

    void clear() override { NOT_IMPLEMENTED; }

    Cursor<Node> children() override
    {
        NOT_IMPLEMENTED;
        return {};
    }

    Cursor<const Node> children() const override
    {
        NOT_IMPLEMENTED;
        return {};
    }

    void for_each(std::function<void(const Node&, Node&)> const&) override { NOT_IMPLEMENTED; }

    void for_each(std::function<void(const Node&, const Node&)> const&) const override { NOT_IMPLEMENTED; }

    //----------------

    Node update(const Node&, const Node& = {}, const Node& opt = {}) override
    {
        NOT_IMPLEMENTED;
        return Node{};
    }

    Node fetch(const Node&, const Node& projection = {}, const Node& opt = {}) const override
    {
        NOT_IMPLEMENTED;
        return Node{};
    }

    //----------------

    bool contain(const std::string&) const override { return false; }

    void update_child(const std::string&, const Node&) override { NOT_IMPLEMENTED; }

    Node insert_child(const std::string&, const Node&) override
    {
        NOT_IMPLEMENTED;
        return Node{};
    }

    Node find_child(const std::string&) const override
    {
        NOT_IMPLEMENTED;
        return Node{};
    }

    void remove_child(const std::string&) override { NOT_IMPLEMENTED; }
}; // namespace sp::db

std::shared_ptr<NodeObject> create_node_object(const Node& opt);

#define SPDB_ENTRY_REGISTER(_NAME_, _CLASS_)               \
    template <>                                            \
    bool ::sp::db::NodePlugin<_CLASS_>::is_registered =    \
        ::sp::utility::Factory<::sp::db::NodeObject>::add( \
            __STRING(_NAME_),                              \
            []() { return dynamic_cast<::sp::db::NodeObject*>(new ::sp::db::NodePlugin<_CLASS_>()); });

#define SPDB_ENTRY_ASSOCIATE(_NAME_, _CLASS_, ...)      \
    template <>                                         \
    int ::sp::db::NodePlugin<_CLASS_>::associated_num = \
        ::sp::utility::Factory<::sp::db::NodeObject>::associate(__STRING(_NAME_), __VA_ARGS__);

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_