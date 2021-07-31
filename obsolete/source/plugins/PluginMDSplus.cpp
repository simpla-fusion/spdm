#include "../db/NodePlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>

namespace sp::db
{
struct mdsplus_node
{
};
typedef NodePlugin<mdsplus_node> NodePluginMDSPLUS;

template <>
void NodePluginMDSPLUS::load(const Node& uri)
{
    VERBOSE << "Load MDSPLUS document :" << uri;
}

template <>
void NodePluginMDSPLUS::save(const Node& url) const
{
}

template <>
void NodePluginMDSPLUS::clear() { NOT_IMPLEMENTED; }

template <>
Cursor<const Node>
NodePluginMDSPLUS::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

template <>
Cursor<Node>
NodePluginMDSPLUS::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

template <>
void NodePluginMDSPLUS::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
}

template <>
Node NodePluginMDSPLUS::update(const Node&, const Node&, const Node& opt)
{
    Node res;
    NOT_IMPLEMENTED;
    return std::move(res);
}

template <>
Node NodePluginMDSPLUS::fetch(const Node&, const Node& projection, const Node& opt) const
{
    NOT_IMPLEMENTED;
    return Node{};
}

template <>
void NodePluginMDSPLUS::update_child(const std::string& path, const Node& v) { NOT_IMPLEMENTED; }

template <>
Node NodePluginMDSPLUS::insert_child(const std::string& path, const Node& v)
{
    NOT_IMPLEMENTED;
    return Node{};
}
template <>
Node NodePluginMDSPLUS::find_child(const std::string& path) const
{
    NOT_IMPLEMENTED;
    return Node{};
}

template <>
void NodePluginMDSPLUS::remove_child(const std::string& name) { NOT_IMPLEMENTED; }

SPDB_ENTRY_REGISTER(mdsplus, mdsplus_node);
SPDB_ENTRY_ASSOCIATE(mdsplus, mdsplus_node, "^(.*)\\.(mdsplus)$");

} // namespace sp::db