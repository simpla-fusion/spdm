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
    std::shared_ptr<pugi::xml_document> doc;
    std::variant<pugi::xml_node, std::string> node;
};
typedef NodePlugin<xml_node> NodePluginXML;

//----------------------------------------------------------------------------------

void load_xml_from_file(pugi::xml_document* doc, std::string const& path)
{
    auto result = doc->load_file(path.c_str());

    if (!result)
    {
        RUNTIME_ERROR << result.description();
        throw std::runtime_error("Can not load file: " + path);
    }
    else
    {
        VERBOSE << "Load XML document: " << path;
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
        VERBOSE << "Load XML document :" << content.substr(0, 20) << "...";
    }
}

std::string join_path(std::string const& prefix, std::string const& path)
{
    std::string res;
    if (path[0] == '/')
    {
        res = path;
    }
    else
    {
        auto pos = prefix.rfind('/');
        if (pos != std::string::npos)
        {
            res = prefix.substr(0, pos) + "/" + path;
        }
        else
        {
            res = path;
        }
    }
    return res;
}

bool process_xinclude(pugi::xml_node& node, std::string const& prefix = "")
{
    for (pugi::xml_node child = node.first_child(); child;)
    {

        pugi::xml_node current = child;

        child = child.next_sibling();

        if (current.type() == pugi::node_element && (strcmp(current.name(), "xi:include") == 0))
        {
            pugi::xml_document external_doc;

            std::string path = join_path(prefix, current.attribute("href").as_string());

            auto result = external_doc.load_file(path.c_str(), pugi::parse_default | pugi::parse_pi); // for <?include?>

            if (result)
            {
                VERBOSE << "Load xml file [" << path << "] (" << result.description() << ")";

                node.insert_child_before(pugi::node_comment, current).set_value(path.c_str());

                // copy the document above the include directive (this retains the original order!)
                for (auto&& ic : external_doc.children())
                {
                    node.insert_copy_before(ic, current);
                }
                // remove the include node and move to the next child
                node.remove_child(current);
            }
            else
            {
                VERBOSE << "Can not load external xml file: " << path;
            }
        }
        else
        {
            process_xinclude(current, prefix);
        }
    }

    return true;
}

template <>
void NodePluginXML::load(const Node& opt)
{
    m_container_.doc = std::make_shared<pugi::xml_document>();
    m_container_.node.emplace<std::string>("");

    std::string file_path = "";

    std::visit(
        traits::overloaded{
            [&](const std::variant_alternative_t<Node::tags::Path, Node::value_type>& path) {
                file_path = path.str();
                load_xml_from_file(m_container_.doc.get(), file_path);
            },
            [&](const std::variant_alternative_t<Node::tags::String, Node::value_type>& str) {
                if (str.substr(7) == "```xml{")
                {
                    load_xml_from_string(m_container_.doc.get(), str.substr(7, str.size() - 1));
                }
                else
                {
                    file_path = str;
                    load_xml_from_file(m_container_.doc.get(), file_path);
                }
            },
            [&](const std::variant_alternative_t<Node::tags::Object, Node::value_type>& obj) {
                auto content = obj->find_child("content");
                if (!content.is_null())
                {
                    load_xml_from_string(m_container_.doc.get(), content.get_value<std::string>());
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

    if (m_container_.doc != nullptr)
    {
        process_xinclude(*m_container_.doc, file_path);
    }
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

const char* node_types[] = {"null", "document", "element", "pcdata", "cdata", "comment", "pi", "declaration"};

// Node make_node(const pugi::xml_node& n)
// {
//     Node res;

//     // VERBOSE << n.path() << ":" << node_types[n.type()];

//     switch (std::distance(n.children().begin(), n.children().end()))
//     {
//     case 0:
//         break;
//     case 1:
//         if (n.first_child().type() == pugi::xml_node_type::node_pcdata)
//         {
//             res.set_value<Node::tags::String>(n.first_child().value());
//             break;
//         }
//     default:
//     {
//         auto r = std::make_shared<NodePluginXML>();
//         r->container().emplace<0>(n);
//         res.set_value<Node::tags::Object>(r);
//     }
//     }

//     return std::move(res);
// }

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

Node make_node(const xml_node& holder, const pugi::xml_node& node)
{
    Node res;

    switch (std::distance(node.children().begin(), node.children().end()))
    {
    case 0:
        break;
    case 1:
        if (node.first_child().type() == pugi::xml_node_type::node_pcdata)
        {
            res.set_value<Node::tags::String>(node.first_child().value());
            break;
        }
    default:
        res.set_value<Node::tags::Object>(std::make_shared<NodePluginXML>(xml_node{holder.doc, node}));
    }
    return res;
}

Node make_node(const xml_node& holder, const std::string& xpath)
{
    auto node_set = holder.doc->select_nodes(xpath.c_str());

    Node res;

    switch (std::distance(node_set.begin(), node_set.end()))
    {
    case 0:
        break;
    case 1:
        make_node(holder, node_set.begin()->node()).swap(res);
        break;
    default:
        auto array = NodeArray::create();
        for (auto&& n : node_set)
        {
            array->push_back(Node{std::make_shared<NodePluginXML>(xml_node{holder.doc, n.node()})});
        }
        res.set_value<Node::tags::Array>(array);
        break;
    }
    return std::move(res);
}
template <>
void NodePluginXML::for_each(std::function<void(const Node&, const Node&)> const& visitor) const
{
    std::visit(
        traits::overloaded{
            [&](const pugi::xml_node& node) {
                for (auto&& n : node.children())
                {
                    visitor(n.name(), make_node(m_container_, n));
                }
            },
            [&](const std::string& xpath) {
                if (xpath == "")
                {
                    for (auto&& n : m_container_.doc->children())
                    {
                        visitor(n.name(), make_node(m_container_, n));
                    }
                }
                else
                {
                    for (auto&& n : m_container_.doc->select_nodes(xpath.c_str()))
                    {
                        visitor(n.node().name(), make_node(m_container_, n.node()));
                    }
                }
            },
            [&](auto&&) {}},
        m_container_.node);
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
            [&](const pugi::xml_node& node) {
                NOT_IMPLEMENTED;
            },
            [&](const std::string& path) {
                query.visit(
                    traits::overloaded{
                        [&](const std::variant_alternative_t<sp::db::Node::tags::String, sp::db::Node::value_type>& uri) {
                            make_node(m_container_, uri).swap(res);
                        },
                        [&](const std::variant_alternative_t<sp::db::Node::tags::Path, sp::db::Node::value_type>& path) {
                            make_node(m_container_, path_to_xpath(path)).swap(res);
                        },
                        [&](const std::variant_alternative_t<sp::db::Node::tags::Object, sp::db::Node::value_type>& object_p) {
                            NOT_IMPLEMENTED;
                        },
                        [&](auto&& ele) { NOT_IMPLEMENTED; } //
                    });
            },
            [&](auto&&) {}},
        m_container_.node);

    return res;
}

template <>
Node NodePluginXML::find_child(const std::string& name) const { return fetch(name); }

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