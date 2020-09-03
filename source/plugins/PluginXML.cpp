#include "../db/Cursor.h"
#include "../db/Entry.h"
#include "../db/Node.h"
#include "../db/NodePlugin.h"
#include "../db/XPath.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include "pugixml/pugixml.hpp"
#include <variant>
namespace sp::db
{
struct xml_node
{
    std::shared_ptr<pugi::xml_node> root;
    std::string path;
    std::shared_ptr<pugi::xml_node> node;
};

typedef NodePlugin<xml_node> NodePluginXML;

Node make_node(const pugi::xml_node& node, xml_node const& parent)
{
    VERBOSE << "Type =" << node.type() << "  Text=" << node.text().as_string();

    Node res;

    if (node.empty())
    {
    }
    else if (node.type() == pugi::node_element && !node.text().empty())
    {
        res.as<Node::tags::String>(node.text().as_string());
    }
    else
    {
        Node{std::make_shared<NodePluginXML>(xml_node{parent.root,
                                                      parent.path + "/" + node.name(),
                                                      std::make_shared<pugi::xml_node>(node)})};
    }
    return std::move(res);
}

Node make_node(const pugi::xpath_node_set& nodes, xml_node const& parent)
{
    Node res;

    return std::move(res);
}

std::string path_to_xpath(const Path& path)
{
    std::ostringstream os;

    for (auto&& item : path)
    {
        switch (item.index())
        {
        case Path::tags::Key:
            os << "/" << std::get<Path::tags::Key>(item);
            break;
        case Path::tags::Index:
            os << "[@id=" << std::get<Path::tags::Index>(item) << "]";
            break;
        default:
            NOT_IMPLEMENTED;
            break;
        }
    }

    return os.str();
}
//----------------------------------------------------------------------------------

template <>
NodePluginXML::NodePlugin(const xml_node& container) : m_container_(container) {}

template <>
NodePluginXML::NodePlugin(xml_node&& container) : m_container_(std::move(container)) {}

template <>
NodePluginXML::NodePlugin(const NodePlugin& other) : m_container_{other.m_container_} {}

template <>
NodePluginXML::NodePlugin(NodePlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
void NodePluginXML::load(const Node& opt)
{
    auto uri = opt.as<std::string>();

    VERBOSE << "Load XML document :" << uri;

    auto* doc = new pugi::xml_document;
    auto result = doc->load_file(uri.c_str());
    if (!result)
    {
        RUNTIME_ERROR << result.description();
    }

    m_container_.root = std::shared_ptr<pugi::xml_node>(doc);
    m_container_.path = uri + ":/";
    m_container_.node = m_container_.root;
}

template <>
void NodePluginXML::save(const Node& opt) const
{
    auto url = opt.as<std::string>();

    auto result = reinterpret_cast<pugi::xml_document*>(m_container_.root.get())->save_file(url.c_str());

    if (!result)
    {
        RUNTIME_ERROR << "Write file " << url << " failed!";
    }
    else
    {
        VERBOSE << "Write file " << url << " success!";
    }
}

template <>
void NodePluginXML::clear() { NOT_IMPLEMENTED; }

// template <>
// Entry NodePluginXML::at(Path path) { return Entry{Node{shared_from_this()}, path}; }
// template <>
// Entry NodePluginXML::at(Path path) const { return Entry{Node{const_cast<NodePluginXML*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const Node>
NodePluginXML::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

template <>
Cursor<Node>
NodePluginXML::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

template <>
void NodePluginXML::for_each(std::function<void(const std::string&, const Node&)> const& visitor) const
{
    Node entry;

    for (auto&& attr : m_container_.node->attributes())
    {
        Node v{std::string(attr.value())};
        visitor(std::string("@") + attr.name(), v);
    }

    for (auto&& node : m_container_.node->children())
    {
        visitor(node.name(), make_node(node, m_container_));
    }
}

template <>
Node NodePluginXML::update(const Node&, const Node&, const Node& opt)
{
    NOT_IMPLEMENTED;
    return Node{};
}

template <>
Node NodePluginXML::fetch(const Node&, const Node& projection, const Node& opt) const
{
    NOT_IMPLEMENTED;
    return Node{};
}
//-----------------------------------------------------------------------------------------------------
// as arraytemplate <>
// template <>
// std::shared_ptr<Entry>
// EntryArrayXML::push_back()
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// template <>
// void EntryArrayXML::pop_back()
// {
//     NOT_IMPLEMENTED;
// }

// template <>
// std::shared_ptr<Entry>
// EntryArrayXML::get(int idx)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// template <>
// std::shared_ptr<const Entry>
// EntryArrayXML::get(int idx) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

SPDB_ENTRY_REGISTER(xml, xml_node);
SPDB_ENTRY_ASSOCIATE(xml, xml_node, "^(.*)\\.(xml|xls|xlsx)$");
} // namespace sp::db