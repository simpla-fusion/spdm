#include "../db/Cursor.hpp"
#include "../db/Entry.hpp"
#include "../db/Node.hpp"
#include "../db/NodePlugin.hpp"
#include "../db/XPath.hpp"
#include "../utility/Factory.hpp"
#include "../utility/Logger.hpp"

#include <variant>
namespace sp::db
{

struct proxy_node
{
    std::shared_ptr<NodeObject> mapper;
    std::shared_ptr<NodeObject> data_source;
};

typedef NodePlugin<proxy_node> NodePluginProxy;

template <>
void NodePluginProxy::load(const Node& opt)
{
    std::shared_ptr<NodeObject> obj = nullptr;

    opt.visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Node::tags::String, sp::db::Node::value_type>& request) { obj = NodeObject::create(request); },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) { obj = opt.as<Node::tags::Object>(); },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        });

    if (obj == nullptr)
    {
        NOT_IMPLEMENTED;
    }

    m_container_.mapper = NodeObject::create(obj->find_child("mapper"));
    m_container_.data_source = NodeObject::create(obj->find_child("source"));
}

template <>
void NodePluginProxy::save(const Node& opt) const
{
    NOT_IMPLEMENTED;
}

template <>
Cursor<const Node>
NodePluginProxy::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

template <>
Cursor<Node>
NodePluginProxy::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

template <>
void NodePluginProxy::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
    // m_container_.data_source->for_each(visitor);
    NOT_IMPLEMENTED;
}

template <>
Node NodePluginProxy::update(const Node& query, const Node& patch, const Node& opt)
{
    return m_container_.data_source->update(m_container_.mapper->fetch(query), patch, opt);
}

template <>
Node NodePluginProxy::fetch(const Node& query, const Node& projection, const Node& opt) const
{
    return m_container_.data_source->fetch(m_container_.mapper->fetch(query, projection), opt);
}

template <>
bool NodePluginProxy::contain(const std::string& name) const
{
    return m_container_.data_source->contain(m_container_.mapper->fetch(name).get_value<std::string>());
}

template <>
void NodePluginProxy::update_child(const std::string& name, const Node& d)
{
    return m_container_.data_source->update_child(m_container_.mapper->fetch(name).get_value<std::string>(), d);
}

template <>
Node NodePluginProxy::insert_child(const std::string& name, const Node& d)
{
    return m_container_.data_source->insert_child(m_container_.mapper->fetch(name).get_value<std::string>(), d);
}

template <>
Node NodePluginProxy::find_child(const std::string& name) const
{
    return m_container_.data_source->find_child(m_container_.mapper->fetch(name).get_value<std::string>());
}

template <>
void NodePluginProxy::remove_child(const std::string& name)
{
    m_container_.data_source->remove_child(m_container_.mapper->fetch(name).get_value<std::string>());
}

SPDB_ENTRY_REGISTER(proxy, proxy_node);
SPDB_ENTRY_ASSOCIATE(proxy, proxy_node, "^(proxy:\\/\\/)(.*)$");
} // namespace sp::db