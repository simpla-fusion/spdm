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
    Entry proxy;
    std::shared_ptr<NodeObject> source;
};

typedef NodePlugin<proxy_node> NodePluginProxy;

template <>
NodePluginProxy::NodePlugin(const proxy_node& container) : m_container_(container) {}

template <>
NodePluginProxy::NodePlugin(proxy_node&& container) : m_container_(std::move(container)) {}

template <>
NodePluginProxy::NodePlugin(const NodePlugin& other) : m_container_{other.m_container_} {}

template <>
NodePluginProxy::NodePlugin(NodePlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
std::pair<std::shared_ptr<NodeObject>, Path> NodePluginProxy::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<NodeObject>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const NodeObject>, Path> NodePluginProxy::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const NodeObject>, Path>{nullptr, Path{}};
}

template <>
void NodePluginProxy::load(const tree_node_type& opt)
{
}

template <>
void NodePluginProxy::save(const tree_node_type& opt) const
{
}

template <>
size_t NodePluginProxy::size() const { return 0; }

template <>
void NodePluginProxy::clear() { NOT_IMPLEMENTED; }

template <>
Cursor<const tree_node_type>
NodePluginProxy::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const tree_node_type>{};
}

template <>
Cursor<tree_node_type>
NodePluginProxy::children()
{
    NOT_IMPLEMENTED;
    return Cursor<tree_node_type>{};
}

template <>
void NodePluginProxy::for_each(std::function<void(const std::string&, const tree_node_type&)> const& visitor) const
{
    NOT_IMPLEMENTED;
}

template <>
tree_node_type NodePluginProxy::insert(Path path, tree_node_type v) { return m_container_.source->insert(m_container_.proxy[path], v); }

template <>
void NodePluginProxy::update(Path path, tree_node_type v) { m_container_.source->update(m_container_.proxy[path], v); }

template <>
tree_node_type NodePluginProxy::find(Path path) const
{
    std::ostringstream os;

    for (auto&& item : path)
    {
        switch (item.index())
        {
        case Path::segment_tags::Key:
            os << "/" << std::get<Path::segment_tags::Key>(item);
            break;
        case Path::segment_tags::Index:
            os << "[@id=" << std::get<Path::segment_tags::Index>(item) << "]";
            break;
        default:
            break;
        }
        // std::visit(
        //     sp::traits::overloaded{
        //         [&](std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) { os << "/" << key; },
        //         [&](std::variant_alternative_t<Path::segment_tags::Index, Path::Segment>& idx) { os << "[@id=" << idx << "]"; },
        //         [&](auto&&) {}},
        //     item);
    }
    tree_node_type res;
    VERBOSE << "XPath=" << os.str();

    return res;
}

template <>
void NodePluginProxy::remove(Path path) { NOT_IMPLEMENTED; }

//-----------------------------------------------------------------------------------------------------
// as arraytemplate <>
// template <>
// std::shared_ptr<Entry>
// EntryArrayProxy::push_back()
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// template <>
// void EntryArrayProxy::pop_back()
// {
//     NOT_IMPLEMENTED;
// }

// template <>
// std::shared_ptr<Entry>
// EntryArrayProxy::get(int idx)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// template <>
// std::shared_ptr<const Entry>
// EntryArrayProxy::get(int idx) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

SPDB_ENTRY_REGISTER(proxy, proxy_node);
SPDB_ENTRY_ASSOCIATE(proxy, proxy_node, "^(proxy:\\/\\/)(.*)$");
} // namespace sp::db