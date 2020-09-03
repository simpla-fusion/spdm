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
    std::string path = "";
};

typedef NodePlugin<xml_node> NodePluginXML;

//----------------------------------------------------------------------------------

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
    m_container_.path = "";
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
Node make_node(const pugi::xml_node& n)
{
    Node res;

    if (std::distance(n.children().begin(), n.children().end()) <= 1)
    {
        res.as<Node::tags::String>(n.text().as_string());
    }
    else
    {
        auto r = std::make_shared<NodePluginXML>();
        r->container().root = std::make_shared<pugi::xml_node>(n);
        r->container().path = "";
        res.as<Node::tags::Object>(std::make_shared<NodePluginXML>(r));
    }
    return std::move(res);
}

Node make_node(const pugi::xpath_node_set& node_set)
{
    Node res;
    if (std::distance(node_set.begin(), node_set.end()) == 1)
    {
        make_node(node_set.begin()->node()).swap(res);
    }
    else
    {
        auto array = NodeArray::create();
        for (auto&& n : node_set)
        {
            array->push_back(n.node());
        }
    }
    return std::move(res);
}
template <>
void NodePluginXML::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
    for (auto&& n : m_container_.root->select_nodes(m_container_.path.c_str()))
    {
        visitor(n.node().name(), make_node(n.node()));
    }
}

template <>
Node NodePluginXML::update(const Node&, const Node&, const Node& opt)
{
    NOT_IMPLEMENTED;
    return Node{};
}

template <>
Node NodePluginXML::fetch(const Node& query, const Node& projection, const Node& opt) const
{

    Node res;
    std::visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Node::tags::String, sp::db::Node::value_type>& uri) {
                make_node(m_container_.root->select_nodes((m_container_.path + "/" + uri).c_str())).swap(res);
            },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Path, sp::db::Node::value_type>& path) {
                make_node(m_container_.root->select_nodes((m_container_.path + "/" + path.str()).c_str())).swap(res);
            },
            [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& ele) { NOT_IMPLEMENTED; } //
        },
        query.value());
    return res;
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