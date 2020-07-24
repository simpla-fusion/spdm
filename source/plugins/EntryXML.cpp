#include "../Entry.h"
#include "../EntryCursor.h"
#include "../EntryPlugin.h"
#include "../utility/Logger.h"
#include "pugixml/pugixml.hpp"
#include <variant>
namespace sp
{

struct entry_xml
{
    entry_xml() : m_node_(std::dynamic_pointer_cast<pugi::xml_node>(std::make_shared<pugi::xml_document>())) {}

    entry_xml(const pugi::xml_node& n) : m_node_(std::make_shared<pugi::xml_node>(n)) {}

    entry_xml(pugi::xml_node&& n) : m_node_(std::make_shared<pugi::xml_node>(std::move(n))) {}

    entry_xml(const std::shared_ptr<pugi::xml_node>& p) : m_node_(p) {}

    entry_xml(const std::string& path) : entry_xml()
    {
        auto doc = std::make_shared<pugi::xml_document>();
        doc->load_file(path.c_str());
        m_node_ = std::dynamic_pointer_cast<pugi::xml_node>(doc);
    }

    entry_xml(const entry_xml& other) : m_node_(other.m_node_) {}

    entry_xml(entry_xml&& other) : m_node_(std::move(other.m_node_)) {}

    ~entry_xml() = default;

    std::shared_ptr<pugi::xml_node> m_node_;
};

template <>
Entry::NodeType EntryPlugin<entry_xml>::type() const { return NodeType(m_pimpl_.m_node_->type()); }

//----------------------------------------------------------------------------------
// attributes
template <>
bool EntryPlugin<entry_xml>::has_attribute(const std::string& name) const
{
    std::cout << FILE_LINE_STAMP << name << std::endl;

    return !m_pimpl_.m_node_->attribute(name.c_str()).empty();
}

template <>
Entry::element_t EntryPlugin<entry_xml>::get_attribute_raw(const std::string& name) const
{
    Entry::element_t res;
    std::cout << FILE_LINE_STAMP << name << std::endl;
    res.emplace<std::string>(m_pimpl_.m_node_->attribute(name.c_str()).value());
    return std::move(res);
}

template <>
void EntryPlugin<entry_xml>::set_attribute_raw(const std::string& name, const Entry::element_t& value)
{
    std::cout << FILE_LINE_STAMP << name << ":" << to_string(value) << std::endl;

    m_pimpl_.m_node_->append_attribute(name.c_str()).set_value(to_string(value).c_str());
}

template <>
void EntryPlugin<entry_xml>::remove_attribute(const std::string& name)
{
    m_pimpl_.m_node_->remove_attribute(name.c_str());
}

template <>
std::map<std::string, Entry::element_t> EntryPlugin<entry_xml>::attributes() const
{
    std::map<std::string, Entry::element_t> res{};
    NOT_IMPLEMENTED;
    return std::move(res);
}
//----------------------------------------------------------------------------------
// level 0
template <>
std::string EntryPlugin<entry_xml>::name() const { return m_pimpl_.m_node_->name(); };

// as leaf
template <>
void EntryPlugin<entry_xml>::set_element(const Entry::element_t& v)
{
    m_pimpl_.m_node_->set_value(to_string(v).c_str());
}

template <>
Entry::element_t EntryPlugin<entry_xml>::get_element() const
{
    return Entry::element_t(std::string(m_pimpl_.m_node_->value()));
}

template <>
void EntryPlugin<entry_xml>::set_tensor(const Entry::tensor_t& v)
{
    NOT_IMPLEMENTED;
}

template <>
Entry::tensor_t EntryPlugin<entry_xml>::get_tensor() const
{
    NOT_IMPLEMENTED;
    return Entry::tensor_t{nullptr, typeid(nullptr), {}};
}

template <>
void EntryPlugin<entry_xml>::set_block(const Entry::block_t& v)
{
    NOT_IMPLEMENTED;
}

template <>
Entry::block_t EntryPlugin<entry_xml>::get_block() const
{
    return Entry::block_t{};
}

// as Tree
// as object
template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::find(const std::string& name) const
{
    return std::make_shared<this_type>(m_pimpl_.m_node_->child(name.c_str()));
};

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::find_r(const std::string& path) const
{
    return find(path);
};

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::insert(const std::string& name)
{
    auto n = m_pimpl_.m_node_->child(name.c_str());
    if (n.empty())
    {
        n = m_pimpl_.m_node_->append_child(name.c_str());
    }

    return std::make_shared<this_type>(std::move(n));
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::insert_r(const std::string& path)
{
    // NOT_IMPLEMENTED;
    return insert(path);
}

template <>
void EntryPlugin<entry_xml>::remove(const std::string& name)
{
    NOT_IMPLEMENTED;
}

template <>
size_t EntryPlugin<entry_xml>::size() const
{
    NOT_IMPLEMENTED;

    return 0;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::parent() const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::first_child() const
{
    return std::make_shared<this_type>(m_pimpl_.m_node_->first_child());
}

// as arraytemplate <>
template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::push_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry> EntryPlugin<entry_xml>::pop_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_xml>::item(int idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

SP_REGISTER_ENTRY(xml, entry_xml);

} // namespace sp