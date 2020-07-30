#include "../Entry.h"
#include "../EntryCursor.h"
#include "../EntryPlugin.h"
#include "../utility/Logger.h"
#include "pugixml/pugixml.hpp"
#include <variant>
namespace sp::db
{

template <>
Entry::NodeType EntryPlugin<pugi::xml_node>::type() const { return NodeType(m_pimpl_.type()); }

//----------------------------------------------------------------------------------
// attributes
template <>
bool EntryPlugin<pugi::xml_node>::has_attribute(const std::string& name) const
{
    return !m_pimpl_.attribute(name.c_str()).empty();
}

template <>
Entry::element_t EntryPlugin<pugi::xml_node>::get_attribute_raw(const std::string& name) const
{
    Entry::element_t res;
    res.emplace<std::string>(m_pimpl_.attribute(name.c_str()).value());
    return std::move(res);
}

template <>
void EntryPlugin<pugi::xml_node>::set_attribute_raw(const std::string& name, const Entry::element_t& value)
{
    m_pimpl_.append_attribute(name.c_str()).set_value(to_string(value).c_str());
}

template <>
void EntryPlugin<pugi::xml_node>::remove_attribute(const std::string& name)
{
    m_pimpl_.remove_attribute(name.c_str());
}

template <>
std::map<std::string, Entry::element_t> EntryPlugin<pugi::xml_node>::attributes() const
{
    std::map<std::string, Entry::element_t> res{};
    NOT_IMPLEMENTED;
    return std::move(res);
}
//----------------------------------------------------------------------------------
// level 0
template <>
std::string EntryPlugin<pugi::xml_node>::name() const { return m_pimpl_.name(); };

// as leaf
template <>
void EntryPlugin<pugi::xml_node>::set_element(const Entry::element_t& v)
{
    m_pimpl_.text().set(to_string(v).c_str());
}

template <>
Entry::element_t EntryPlugin<pugi::xml_node>::get_element() const
{
    return Entry::element_t(std::string(m_pimpl_.text().as_string()));
}

template <>
void EntryPlugin<pugi::xml_node>::set_tensor(const Entry::tensor_t& v)
{
    NOT_IMPLEMENTED;
}

template <>
Entry::tensor_t EntryPlugin<pugi::xml_node>::get_tensor() const
{
    NOT_IMPLEMENTED;
    return Entry::tensor_t{nullptr, typeid(nullptr), {}};
}

template <>
void EntryPlugin<pugi::xml_node>::set_block(const Entry::block_t& v)
{
    NOT_IMPLEMENTED;
}

template <>
Entry::block_t EntryPlugin<pugi::xml_node>::get_block() const
{
    return Entry::block_t{};
}

// as Tree
// as object
template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::find(const std::string& name) const
{
    return std::make_shared<this_type>(m_pimpl_.child(name.c_str()));
};

template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::find_r(const std::string& path) const
{
    return find(path);
};

template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::insert(const std::string& name)
{
    auto n = m_pimpl_.child(name.c_str());
    if (n.empty())
    {
        n = m_pimpl_.append_child(name.c_str());
    }

    return std::make_shared<this_type>(n);
}

template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::insert_r(const std::string& path)
{
    // NOT_IMPLEMENTED;
    return insert(path);
}

template <>
void EntryPlugin<pugi::xml_node>::remove(const std::string& name)
{
    NOT_IMPLEMENTED;
}

template <>
size_t EntryPlugin<pugi::xml_node>::size() const
{
    NOT_IMPLEMENTED;

    return 0;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::parent() const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::first_child() const
{
    return std::make_shared<this_type>(m_pimpl_.first_child());
}

// as arraytemplate <>
template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::push_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry> EntryPlugin<pugi::xml_node>::pop_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<pugi::xml_node>::item(int idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

SP_REGISTER_ENTRY(xml, pugi::xml_node);

} // namespace sp::db