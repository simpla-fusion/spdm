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

typedef std::variant<pugi::xml_node, std::pair<std::shared_ptr<pugi::xml_node>, std::string>> xml_node;

typedef NodePlugin<xml_node> NodePluginXML;

//----------------------------------------------------------------------------------

template <>
void NodePluginXML::load(const Node& opt)
{
    if (opt.type() == Node::tags::Null)
    {
        NOT_IMPLEMENTED;
        return;
    }
    else if (opt.type() == Node::tags::String)
    {
        auto uri = opt.as<std::string>();

        VERBOSE << "Load XML document :" << uri;

        auto* doc = new pugi::xml_document;
        auto result = doc->load_file(uri.c_str());
        if (!result)
        {
            RUNTIME_ERROR << result.description();
        }

        m_container_.emplace<1>(std::make_pair(std::shared_ptr<pugi::xml_node>(doc), std::string("")));
    }
}

template <>
void NodePluginXML::save(const Node& opt) const
{
    auto url = opt.as<std::string>();

    // auto result = reinterpret_cast<pugi::xml_document*>(m_container_.root.get())->save_file(url.c_str());

    // if (!result)
    // {
    //     RUNTIME_ERROR << "Write file " << url << " failed!";
    // }
    // else
    // {
    //     VERBOSE << "Write file " << url << " success!";
    // }
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
const char* node_types[] =
    {
        "null", "document", "element", "pcdata", "cdata", "comment", "pi", "declaration"};

Node make_node(const pugi::xml_node& n)
{
    Node res;

    // VERBOSE << n.path() << ":" << node_types[n.type()];

    switch (std::distance(n.children().begin(), n.children().end()))
    {
    case 0:
        break;
    case 1:
        if (n.first_child().type() == pugi::xml_node_type::node_pcdata)
        {
            res.as<Node::tags::String>(n.first_child().value());
            break;
        }
    default:
    {
        auto r = std::make_shared<NodePluginXML>();
        r->container().emplace<0>(n);
        res.as<Node::tags::Object>(r);
    }
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
            array->push_back(make_node(n.node()));
        }
        res.as<Node::tags::Array>(array);
    }
    return std::move(res);
}

template <>
void NodePluginXML::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
    std::visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<0, xml_node>& node) {
                for (auto&& n : node.children())
                {
                    visitor(n.name(), make_node(n));
                }
            },
            [&](const std::variant_alternative_t<1, xml_node>& link) {
                for (auto&& n : link.first->select_nodes(link.second.c_str()))
                {
                    visitor(n.node().name(), make_node(n.node()));
                }
            },
            [&](auto&&) {}},
        m_container_);
}

std::string path_to_xpath(const Path& p)
{
    std::ostringstream os;
    for (const auto& seg : p)
    {
        std::visit(
            traits::overloaded{
                [&](const std::variant_alternative_t<sp::db::Path::tags::Key, sp::db::Path::Segment>& key) { os << "/" << key; },
                [&](const std::variant_alternative_t<sp::db::Path::tags::Index, sp::db::Path::Segment>& idx) { os << "[@id=" << idx << "]"; },
                [&](auto&&) {}},
            seg);
    }
    return os.str();
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
            [&](const std::variant_alternative_t<0, xml_node>& node) {
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<1, xml_node>& link) {
                std::visit(
                    traits::overloaded{
                        [&](const std::variant_alternative_t<sp::db::Node::tags::String, sp::db::Node::value_type>& uri) {
                            make_node(link.first->select_nodes((link.second + "/" + uri).c_str())).swap(res);
                        },
                        [&](const std::variant_alternative_t<sp::db::Node::tags::Path, sp::db::Node::value_type>& path) {
                            make_node(link.first->select_nodes((link.second + path_to_xpath(path)).c_str())).swap(res);
                        },
                        [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) {
                            NOT_IMPLEMENTED;
                        },
                        [&](auto&& ele) { NOT_IMPLEMENTED; } //
                    },
                    query.value());
            },
            [&](auto&&) {}},
        m_container_);

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