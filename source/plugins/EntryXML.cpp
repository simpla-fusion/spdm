#include "../db/Cursor.h"
#include "../db/Entry.h"
#include "../db/EntryPlugin.h"
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

typedef EntryObjectPlugin<xml_node> EntryObjectXML;

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
// void EntryObjectXML::remove(Path const&)
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
// EntryObjectXML::kv_items()
// {
//     NOT_IMPLEMENTED;
//     return Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>{};
// }
// template <>
// Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>
// EntryObjectXML::kv_items() const
// {
//     NOT_IMPLEMENTED;
//     return Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>{};
// }

//----------------------------------------------------------------------------------

template <>
EntryObjectXML::EntryObjectPlugin(const xml_node& container) : m_container_(container) {}

template <>
EntryObjectXML::EntryObjectPlugin(xml_node&& container) : m_container_(std::move(container)) {}

template <>
EntryObjectXML::EntryObjectPlugin(const EntryObjectPlugin& other) : m_container_{other.m_container_} {}

template <>
EntryObjectXML::EntryObjectPlugin(EntryObjectPlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
std::pair<std::shared_ptr<EntryObject>, Path> EntryObjectXML::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<EntryObject>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const EntryObject>, Path> EntryObjectXML::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const EntryObject>, Path>{nullptr, Path{}};
}
template <>
void EntryObjectXML::load(const std::string& uri)
{
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
void EntryObjectXML::save(const std::string& url) const
{
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
size_t EntryObjectXML::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

template <>
void EntryObjectXML::clear()
{
    NOT_IMPLEMENTED;
}

// template <>
// Entry EntryObjectXML::at(const Path& path) { return Entry{entry_value_type{shared_from_this()}, path}; }
// template <>
// Entry EntryObjectXML::at(const Path& path) const { return Entry{entry_value_type{const_cast<EntryObjectXML*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const entry_value_type>
EntryObjectXML::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

template <>
Cursor<entry_value_type>
EntryObjectXML::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>{};
}

// template <>
// void EntryObjectXML::for_each(std::function<void(const std::string&, entry_value_type&)> const& visitor)
// {
//     entry_value_type entry;

//     for (auto&& attr : m_container_.node->attributes())
//     {
//         entry_value_type v{std::string(attr.value())};
//         visitor(std::string("@") + attr.name(), v);
//     }

//     for (auto&& node : m_container_.node->children())
//     {
//         if (node.type() == pugi::node_element)
//         {
//             entry_value_type entry{
//                 std::shared_ptr<EntryObjectXML>(
//                     new EntryObjectXML{
//                         xml_node{
//                             m_container_.root,
//                             m_container_.path + "/" + node.name(),
//                             std::make_shared<pugi::xml_node>(node)}})};

//             visitor(node.name(), entry);
//         }
//         else
//         {
//             entry_value_type entry{std::string(node.value())};

//             visitor(node.name(), entry);
//         }
//     }
// }

template <>
void EntryObjectXML::for_each(std::function<void(const std::string&, const entry_value_type&)> const& visitor) const
{
    entry_value_type entry;

    for (auto&& attr : m_container_.node->attributes())
    {
        entry_value_type v{std::string(attr.value())};
        visitor(std::string("@") + attr.name(), v);
    }

    for (auto&& node : m_container_.node->children())
    {
        if (node.type() == pugi::node_element && node)
        {
            visitor(node.name(),
                    entry_value_type{
                        std::shared_ptr<EntryObjectXML>(
                            new EntryObjectXML{
                                xml_node{
                                    m_container_.root,
                                    m_container_.path + "/" + node.name(),
                                    std::make_shared<pugi::xml_node>(node)}})});
        }
        else
        {
            visitor(node.name(), entry_value_type{std::string(node.value())});
        }
    }
}

template <>
entry_value_type EntryObjectXML::insert(const std::string& key, entry_value_type v)
{

    entry_value_type res;
    // res->emplace<Entry::type_tags::Object>(std::dynamic_pointer_cast<EntryObject>(std::make_shared<this_type>(self(), xml_node{nullptr, n})));
    return std::move(res);
}
template <>
entry_value_type EntryObjectXML::find(const std::string& key) const
{
    // NOT_IMPLEMENTED;
    // if (m_container_.document == nullptr)
    // {
    //     m_container_.document = std::make_shared<pugi::xml_document>();
    //     m_container_.document->load_file(path.str().c_str());
    //     m_container_.node = m_container_.document->root();
    // }
    return entry_value_type{};
}

template <>
void EntryObjectXML::update(const std::string& key, entry_value_type v) {}

template <>
void EntryObjectXML::remove(const std::string& path) {}

template <>
entry_value_type EntryObjectXML::insert(const Path& path, entry_value_type v) { return EntryObject::insert(path, std::move(v)); }

template <>
void EntryObjectXML::update(const Path& path, entry_value_type v) { EntryObject::update(path, std::move(v)); }

template <>
entry_value_type EntryObjectXML::find(const Path& path) const { return EntryObject::find(path); }

template <>
void EntryObjectXML::remove(const Path& path) { EntryObject::remove(path); }

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