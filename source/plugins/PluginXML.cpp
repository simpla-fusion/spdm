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

tree_node_type make_entry(const pugi::xml_node& node, xml_node const& parent)
{
    VERBOSE << "Type =" << node.type() << "  Text=" << node.text().as_string();

    tree_node_type res;

    if (node.empty())
    {
    }
    else if (node.type() == pugi::node_element && !node.text().empty())
    {
        res.emplace<tree_node_tags::String>(node.text().as_string());
    }
    else
    {
        tree_node_type{std::make_shared<NodePluginXML>(xml_node{parent.root,
                                                                parent.path + "/" + node.name(),
                                                                std::make_shared<pugi::xml_node>(node)})};
    }
    return std::move(res);
}
tree_node_type make_entry(const pugi::xpath_node_set& nodes, xml_node const& parent)
{
    tree_node_type res;

    return std::move(res);
}
// template <>
// class CursorProxy<Entry, pugi::xml_node> : public CursorProxy<Entry>
// {
// public:
//     typedef CursorProxy<Entry> base_type;
//     typedef CursorProxy<Entry, pugi::xml_node> this_type;
//     // using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;
//     CursorProxy(const pugi::xml_node& obj) : m_base_(obj), m_pointer_(nullptr) { update(); }
//     CursorProxy(const this_type& other) : m_base_(other.m_base_), m_pointer_(nullptr) { update(); }
//     CursorProxy(this_type&& other) : m_base_(std::move(other.m_base_)), m_pointer_(nullptr) { update(); }
//     virtual ~CursorProxy(){};
//     std::unique_ptr<base_type> copy() const override { return std::unique_ptr<base_type>(new this_type(*this)); }
//     bool done() const override { return m_base_.empty(); }
//     pointer get_pointer() override { return pointer(m_pointer_.get()); }
//     reference get_reference() override { return *m_pointer_; }
//     bool next() override
//     {
//         m_base_ = m_base_.next_sibling();
//         return m_base_.empty();
//     }
// protected:
//     pugi::xml_node m_base_;
//     std::shared_ptr<Entry> m_pointer_;
//     void update()
//     {
//         if (m_base_.empty())
//         {
//             m_pointer_ = (nullptr);
//         }
//         else
//         {
//             m_pointer_ = make_entry(m_base_);
//         }
//     }
// };
// template <>
// class CursorProxy<const Entry, pugi::xml_node> : public CursorProxy<const Entry>
// {
// public:
//     typedef CursorProxy<const Entry> base_type;
//     typedef CursorProxy<const Entry, pugi::xml_node> this_type;
//     // using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;
//     CursorProxy(const pugi::xml_node& obj) : m_base_(obj), m_pointer_(nullptr) { update(); }
//     CursorProxy(const this_type& other) : m_base_(other.m_base_), m_pointer_(nullptr) { update(); }
//     CursorProxy(this_type&& other) : m_base_(std::move(other.m_base_)), m_pointer_(nullptr) { update(); }
//     virtual ~CursorProxy(){};
//     std::unique_ptr<base_type> copy() const override { return std::unique_ptr<base_type>(new this_type(*this)); }
//     bool done() const override { return m_base_.empty(); }
//     pointer get_pointer() override { return pointer(m_pointer_.get()); }
//     reference get_reference() override { return *m_pointer_; }
//     bool next() override
//     {
//         m_base_ = m_base_.next_sibling();
//         return m_base_.empty();
//     }
// protected:
//     pugi::xml_node m_base_;
//     std::shared_ptr<Entry> m_pointer_;
//     void update()
//     {
//         if (m_base_.empty())
//         {
//             m_pointer_ = (nullptr);
//         }
//         else
//         {
//             m_pointer_ = make_entry(m_base_);
//         }
//     }
// };

// template <>
// void NodePluginXML::remove(Path const&)
// {
//     NOT_IMPLEMENTED;
// }
// template <>
// class CursorProxy<std::pair<const std::string, std::shared_ptr<Entry>>,
//                   pugi::xml_node_iterator> : public CursorProxy<std::pair<const std::string, std::shared_ptr<Entry>>>
// {
// public:
//     typedef CursorProxy<std::pair<const std::string, std::shared_ptr<Entry>>> base_type;
//     typedef CursorProxy<std::pair<const std::string, std::shared_ptr<Entry>>, pugi::xml_node_iterator> this_type;
//     typedef pugi::xml_node_iterator iterator;
//     using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;
//     CursorProxy(const iterator& ib, const iterator& ie) : m_it_(ib), m_ie_(ie) {}
//     ~CursorProxy() = default;
//     std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); }
//     bool done() const override { return m_it_ == m_ie_; }
//     reference get_reference() override { return m_mapper_(*m_it_); }
//     bool next() override
//     {
//         if (m_it_ != m_ie_)
//         {
//             ++m_it_;
//         }
//         return !done();
//     }
// protected:
//     iterator m_it_, m_ie_;
// };
// template <>
// Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>
// NodePluginXML::kv_items()
// {
//     NOT_IMPLEMENTED;
//     return Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>{};
// }
// template <>
// Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>
// NodePluginXML::kv_items() const
// {
//     NOT_IMPLEMENTED;
//     return Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>{};
// }

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
std::pair<std::shared_ptr<NodeObject>, Path> NodePluginXML::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<NodePlugin>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const NodeObject>, Path> NodePluginXML::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const NodePlugin>, Path>{nullptr, Path{}};
}

template <>
void NodePluginXML::load(const tree_node_type& opt)
{
    auto uri = std::get<std::string>(opt);

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
void NodePluginXML::save(const tree_node_type& opt) const
{
    auto url = std::get<std::string>(opt);

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
size_t NodePluginXML::size() const { return std::distance(m_container_.node->children().begin(), m_container_.node->children().end()); }

template <>
void NodePluginXML::clear() { NOT_IMPLEMENTED; }

// template <>
// Entry NodePluginXML::at(Path path) { return Entry{tree_node_type{shared_from_this()}, path}; }
// template <>
// Entry NodePluginXML::at(Path path) const { return Entry{tree_node_type{const_cast<NodePluginXML*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const tree_node_type>
NodePluginXML::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const tree_node_type>{};
}

template <>
Cursor<tree_node_type>
NodePluginXML::children()
{
    NOT_IMPLEMENTED;
    return Cursor<tree_node_type>{};
}

template <>
void NodePluginXML::for_each(std::function<void(const std::string&, const tree_node_type&)> const& visitor) const
{
    tree_node_type entry;

    for (auto&& attr : m_container_.node->attributes())
    {
        tree_node_type v{std::string(attr.value())};
        visitor(std::string("@") + attr.name(), v);
    }

    for (auto&& node : m_container_.node->children())
    {
        visitor(node.name(), make_entry(node, m_container_));
    }
}

template <>
tree_node_type NodePluginXML::insert(Path path, tree_node_type v)
{
    NOT_IMPLEMENTED;
    return tree_node_type{};
}

template <>
void NodePluginXML::update(Path path, tree_node_type v) { NOT_IMPLEMENTED; }

template <>
tree_node_type NodePluginXML::find(Path path) const
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

    VERBOSE << "XPath=" << os.str();

    auto node_set = m_container_.node->select_nodes(os.str().c_str());

    tree_node_type res;

    if (node_set.size() == 0)
    {
    }
    else if (node_set.size() == 1)
    {
        make_entry(node_set.begin()->node(), m_container_).swap(res);
    }
    else
    {
        make_entry(node_set, m_container_).swap(res);
    }

    return res;
}

template <>
void NodePluginXML::remove(Path path) { NodePlugin::remove(path); }

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