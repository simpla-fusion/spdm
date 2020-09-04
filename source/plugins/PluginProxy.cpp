#include "../db/Cursor.h"
#include "../db/Entry.h"
#include "../db/Node.h"
#include "../db/NodePlugin.h"
#include "../db/XPath.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"

#include <variant>
namespace sp::db
{

struct proxy_node
{
    std::shared_ptr<NodeObject> mapper;
    std::shared_ptr<NodeObject> source;
};

typedef NodePlugin<proxy_node> NodePluginProxy;

template <>
void NodePluginProxy::load(const Node& opt)
{
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
    NOT_IMPLEMENTED;
}

Node NodePluginProxy::update(const Node& query, const Node& patch, const Node& opt)
{
    return m_container_.source->update(m_container_.mapper->fetch(query), patch, opt);
}

Node NodePluginProxy::fetch(const Node& query, const Node& projection, const Node& opt) const
{
    return m_container_.source->fetch(m_container_.mapper->fetch(query), projection, opt);
}
bool NodePluginProxy::contain(const std::string& name) const
{
    return m_container_.source->contain(m_container_.mapper->fetch(name).get_value<std::string>());
}

void NodePluginProxy::update_child(const std::string& name, const Node& d)
{
    return m_container_.source->update_child(m_container_.mapper->fetch(name).get_value<std::string>(), d);
}

Node NodePluginProxy::insert_child(const std::string& name, const Node& d)
{
    return m_container_.source->insert_child(m_container_.mapper->fetch(name).get_value<std::string>(), d);
}

Node NodePluginProxy::find_child(const std::string&name) const
{
      return m_container_.source->find_child(m_container_.mapper->fetch(name).get_value<std::string>() );

}

void NodePluginProxy::remove_child(const std::string&) { NOT_IMPLEMENTED; }

SPDB_ENTRY_REGISTER(proxy, proxy_node);
SPDB_ENTRY_ASSOCIATE(proxy, proxy_node, "^(proxy:\\/\\/)(.*)$");
} // namespace sp::db