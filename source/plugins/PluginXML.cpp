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

void load_xml_from_file(pugi::xml_document* doc, std::string const& path)
{
    auto result = doc->load_file(path.c_str());

    if (!result)
    {
        RUNTIME_ERROR << result.description();
    }
    else
    {
        VERBOSE << "Load XML document :" << path;
    }
}

void load_xml_from_string(pugi::xml_document* doc, std::string const& content)
{
    auto result = doc->load_string(content.c_str());

    if (!result)
    {
        RUNTIME_ERROR << result.description();
    }
    else
    {
        VERBOSE << "Load XML document :" << content.substr(0,20) << "...";
    }
}
template <>
void NodePluginXML::load(const Node& opt)
{

    auto* doc = new pugi::xml_document;

    m_container_.emplace<1>(std::make_pair(std::shared_ptr<pugi::xml_node>(doc), std::string("")));

    std::visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& path) {
                load_xml_from_file(doc, path.str());
            },
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& str) {
                if (str.substr(7) == "```xml{")
                {
                    load_xml_from_string(doc, str.substr(7, str.size() - 1));
                }
                else
                {
                    load_xml_from_file(doc, str);
                }
            },
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& obj) {
                auto content = obj->find_child("content");
                if (!content.is_null())
                {
                    load_xml_from_string(doc, content.get_value<std::string>());
                }
                else
                {
                    RUNTIME_ERROR << "illegal configuration! plugin xml :" << opt;
                }
            },
            [&](const std::variant_alternative_t<Node::tags::Null, Node::value_type>&) {},
            [&](auto&&) { NOT_IMPLEMENTED; }

        },
        opt.value());
}

template <>
void NodePluginXML::save(const Node& opt) const
{
    auto url = opt.get_value<std::string>();

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
            res.set_value<Node::tags::String>(n.first_child().value());
            break;
        }
    default:
    {
        auto r = std::make_shared<NodePluginXML>();
        r->container().emplace<0>(n);
        res.set_value<Node::tags::Object>(r);
    }
    }

    return std::move(res);
}

std::string path_to_xpath(const Path& p)
{
    std::ostringstream os;
    for (const auto& seg : p)
    {
        std::visit(
            traits::overloaded{
                [&](const std::variant_alternative_t<sp::db::Path::tags::Key, sp::db::Path::Segment>& key) {
                    if (key[0] == '@')
                    {
                        os << "[" << key << "]";
                    }
                    else
                    {
                        os << "/" << key;
                    }
                },
                [&](const std::variant_alternative_t<sp::db::Path::tags::Index, sp::db::Path::Segment>& idx) {
                    if (idx < 0)
                    {
                        os << "[last()" << idx << "]";
                    }
                    else
                    {
                        /**
                         * @NOTE:  according to W3C , pugi::xpath  first node is [1], 
                         *    but in SpDB::Path first node is [0]
                        */
                        os << "[" << idx + 1 << "]";
                    }
                },
                [&](auto&&) {}},
            seg);
    }
    return os.str();
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
        res.set_value<Node::tags::Array>(array);
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