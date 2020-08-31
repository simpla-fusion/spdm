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

class NodeBackend
{

public:
    NodeBackend() = default;
    virtual ~NodeBackend() = default;
    NodeBackend(const NodeBackend&) = delete;
    NodeBackend(NodeBackend&&) = delete;

    static std::shared_ptr<NodeBackend> create(const NodeObject& opt = {});

    virtual std::shared_ptr<NodeBackend> copy() const = 0;

    virtual void load(const NodeObject&) = 0;

    virtual void save(const NodeObject&) const = 0;

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    virtual Cursor<Node> children() = 0;

    virtual Cursor<const Node> children() const = 0;

    virtual void for_each(std::function<void(const std::string&, const Node&)> const&) const = 0;

    virtual void set_value(const std::string& name, Node v) = 0;

    virtual Node get_value(const std::string& name) const = 0;

    virtual Node fetch(const NodeObject&, const NodeObject& opt) const = 0;

    virtual void update(const NodeObject&, const NodeObject& opt) = 0;
};

template <typename Container>
class NodePlugin : public NodeBackend
{
private:
    Container m_container_;
    static bool is_registered;
    static int associated_num;

public:
    typedef NodePlugin<Container> this_type;

    NodePlugin() = default;

    virtual ~NodePlugin() = default;

    NodePlugin(const NodeObject& opt) { load(opt); };

    NodePlugin(const Container& container) : m_container_(container) {}

    NodePlugin(Container&& container) : m_container_(std::move(container)) {}

    NodePlugin(const this_type& other) : m_container_(other.m_container_) {}

    NodePlugin(this_type&& other) : m_container_(std::move(other.m_container_)) {}

    std::shared_ptr<NodeBackend> copy() const override { return std::shared_ptr<NodeBackend>(new this_type(*this)); }

    void load(const NodeObject&) override { NOT_IMPLEMENTED; }

    void save(const NodeObject&) const override { NOT_IMPLEMENTED; }

    size_t size() const override { return m_container_.size(); }

    void clear() override { m_container_.clear(); }

    Cursor<Node> children() override
    {
        NOT_IMPLEMENTED;
        return Cursor<Node>{};
    }

    Cursor<const Node> children() const override
    {
        NOT_IMPLEMENTED;
        return Cursor<const Node>{};
    };

    void for_each(std::function<void(const std::string&, const Node&)> const&) const override { NOT_IMPLEMENTED; }

    void set_value(const std::string& name, Node v) override { NOT_IMPLEMENTED; };

    Node get_value(const std::string& name) const override
    {
        NOT_IMPLEMENTED;
        return Node{};
    };

    void update(const NodeObject&, const NodeObject& opt = {}) override { NOT_IMPLEMENTED; }

    Node fetch(const NodeObject&, const NodeObject& opt = {}) const override
    {
        NOT_IMPLEMENTED;
        return Node{};
    }
};

#define SPDB_ENTRY_REGISTER(_NAME_, _CLASS_)                \
    template <>                                             \
    bool ::sp::db::NodePlugin<_CLASS_>::is_registered =     \
        ::sp::utility::Factory<::sp::db::NodeBackend>::add( \
            __STRING(_NAME_),                               \
            []() { return dynamic_cast<::sp::db::NodeBackend*>(new ::sp::db::NodePlugin<_CLASS_>()); });

#define SPDB_ENTRY_ASSOCIATE(_NAME_, _CLASS_, ...)      \
    template <>                                         \
    int ::sp::db::NodePlugin<_CLASS_>::associated_num = \
        ::sp::utility::Factory<::sp::db::NodeBackend>::associate(__STRING(_NAME_), __VA_ARGS__);

} // namespace sp::db
#endif // SPDB_ENTRY_PLUGIN_H_