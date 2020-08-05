#include "../db/Cursor.h"
#include "../db/Entry.h"
#include "../db/EntryPlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include "pugixml/pugixml.hpp"
#include <variant>
namespace sp::db
{
typedef EntryObjectPlugin<pugi::xml_node> EntryObjectXML;
typedef EntryArrayPlugin<pugi::xml_node> EntryArrayXML;

std::shared_ptr<Entry> make_entry(const pugi::xml_node& node)
{
    return nullptr;
}

template <>
class CursorProxy<Entry, pugi::xml_node> : public CursorProxy<Entry>
{
public:
    typedef CursorProxy<Entry> base_type;
    typedef CursorProxy<Entry, pugi::xml_node> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(const pugi::xml_node& obj) : m_base_(obj), m_pointer_(nullptr) { update(); }

    CursorProxy(const this_type& other) : m_base_(other.m_base_), m_pointer_(nullptr) { update(); }

    CursorProxy(this_type&& other) : m_base_(std::move(other.m_base_)), m_pointer_(nullptr) { update(); }

    virtual ~CursorProxy(){};

    std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const { return m_base_.empty(); }

    pointer get_pointer() override { return pointer(m_pointer_.get()); }

    reference get_reference() override { return *m_pointer_; }

    bool next() override
    {
        m_base_ = m_base_.next_sibling();
        return m_base_.empty();
    }

protected:
    pugi::xml_node m_base_;
    std::shared_ptr<Entry> m_pointer_;

    void update()
    {
        if (m_base_.empty())
        {
            m_pointer_ = (nullptr);
        }
        else
        {
            m_pointer_ = make_entry(m_base_);
        }
    }
};

template <>
class CursorProxy<const Entry, pugi::xml_node> : public CursorProxy<const Entry>
{
public:
    typedef CursorProxy<const Entry> base_type;
    typedef CursorProxy<const Entry, pugi::xml_node> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(const pugi::xml_node& obj) : m_base_(obj), m_pointer_(nullptr) { update(); }

    CursorProxy(const this_type& other) : m_base_(other.m_base_), m_pointer_(nullptr) { update(); }

    CursorProxy(this_type&& other) : m_base_(std::move(other.m_base_)), m_pointer_(nullptr) { update(); }

    virtual ~CursorProxy(){};

    std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const { return m_base_.empty(); }

    pointer get_pointer() override { return pointer(m_pointer_.get()); }

    reference get_reference() override { return *m_pointer_; }

    bool next() override
    {
        m_base_ = m_base_.next_sibling();
        return m_base_.empty();
    }

protected:
    pugi::xml_node m_base_;
    std::shared_ptr<Entry> m_pointer_;

    void update()
    {
        if (m_base_.empty())
        {
            m_pointer_ = (nullptr);
        }
        else
        {
            m_pointer_ = make_entry(m_base_);
        }
    }
};

//----------------------------------------------------------------------------------

template <>
void EntryObjectXML::clear()
{
}
template <>
Cursor<Entry> EntryObjectXML::select(XPath const&)
{
    return Cursor<Entry>{};
}

template <>
Cursor<const Entry> EntryObjectXML::select(XPath const&) const
{
    return Cursor<const Entry>{};
}

template <>
void EntryObjectXML::erase(XPath const&)
{
}

template <>
Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>
EntryObjectXML::kv_items()
{
    return Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>{};
}
template <>
Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>
EntryObjectXML::kv_items() const
{
    return Cursor<std::pair<const std::string, std::shared_ptr<Entry>>>{};
}

template <>
std::shared_ptr<const Entry>
EntryObjectXML::get(const std::string& name) const
{
    return make_entry(m_container_.child(name.c_str()));
};

template <>
std::shared_ptr<const Entry>
EntryObjectXML::get(const XPath& path) const
{
    return get(path);
};

template <>
std::shared_ptr<Entry>
EntryObjectXML::insert(const std::string& name)
{
    auto n = m_container_.child(name.c_str());
    if (n.empty())
    {
        n = m_container_.append_child(name.c_str());
    }
    auto res = std::make_shared<Entry>();
    res->emplace<Entry::type_tags::Object>(std::dynamic_pointer_cast<EntryObject>(std::make_shared<this_type>(self(), n)));
    return res;
}

template <>
std::shared_ptr<Entry>
EntryObjectXML::insert(const XPath& path)
{
    // NOT_IMPLEMENTED;
    return insert(path.str());
}

template <>
void EntryObjectXML::erase(const std::string& name)
{
    NOT_IMPLEMENTED;
}

template <>
size_t EntryObjectXML::size() const
{
    NOT_IMPLEMENTED;

    return 0;
}

template <>
Cursor<const Entry>
EntryObjectXML::children() const { return Cursor<const Entry>(m_container_.first_child()); }

template <>
Cursor<Entry>
EntryObjectXML::children() { return Cursor<Entry>(m_container_.first_child()); }

//-----------------------------------------------------------------------------------------------------
// as arraytemplate <>
template <>
std::shared_ptr<Entry>
EntryArrayXML::push_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
void EntryArrayXML::pop_back()
{
    NOT_IMPLEMENTED;
}
template <>
std::shared_ptr<Entry>
EntryArrayXML::get(int idx)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Entry>
EntryArrayXML::get(int idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

SPDB_ENTRY_REGISTER(xml, pugi::xml_node);
SPDB_ENTRY_ASSOCIATE(xml, pugi::xml_node, "[.*].xml");
} // namespace sp::db