#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
#include "EntryPlugin.h"
namespace sp::db
{

typedef EntryObjectPlugin<std::map<std::string, Entry>> EntryObjectDefault;

//-----------------------------------------------------------------------------------------------------------
Entry::Entry(Entry* parent) : m_parent_(parent) {}

Entry::~Entry() {}

Entry::Entry(const Entry& other) : base_type(other), m_parent_(other.m_parent_) {}

Entry::Entry(Entry&& other) : base_type(std::move(other)), m_parent_(other.m_parent_) {}

void Entry::swap(Entry& other) { base_type::swap(other); }

// Entry& Entry::fetch(const std::string& request)
// {
//     VERBOSE << request << std::endl;

//     Entry* obj = ::sp::utility::Factory<EntryObject, Entry>::create(request, this->shared_from_this());

//     if (obj == nullptr)
//     {
//         RUNTIME_ERROR << "Can not parse request " << request << std::endl;

//         throw std::runtime_error("Can not create Entry for scheme: [" + request + "]");
//     }
//     else
//     {
//         obj->fetch(request);

//         emplace<Entry::type_tags::Object>(obj);

//         VERBOSE << "Load Entry Object plugin:" << request << std::endl;
//     }
//     return *this;
// }

// Entry& Entry::fetch(const XPath& request) { return fetch(request.str()); }

// void Entry::update()
// {
//     if (base_type::index() == type_tags::Reference)
//     {
//         std::get<type_tags::Reference>(*this)->update();
//     }
// }


DataBlock& Entry::as_block()
{
    switch (base_type::index())
    {
    case type_tags::Empty:
        emplace<type_tags::Block>(DataBlock());
        break;
    case type_tags::Block:
        break;
    case type_tags::Reference:
        std::get<type_tags::Reference>(*this)->as_block();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Block>(*this);
}

const DataBlock& Entry::as_block() const
{

    if (type() != type_tags::Block)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Block>(self());
}

EntryObject& Entry::as_object()
{
    switch (base_type::index())
    {
    case type_tags::Empty:
        emplace<type_tags::Object>(std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryObjectDefault>(this->weak_from_this())));
        break;
    case type_tags::Object:
        break;
    case type_tags::Reference:
        std::get<type_tags::Reference>(*this)->as_object();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Object>(self());
}

const EntryObject& Entry::as_object() const
{
    if (type() != type_tags::Object)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Object>(self());
}

EntryArray& Entry::as_array()
{
    switch (base_type::index())
    {
    case type_tags::Empty:
        emplace<type_tags::Array>(std::dynamic_pointer_cast<EntryArray>(std::make_shared<EntryArrayDefault>(this->shared_from_this())));
        break;
    case type_tags::Array:
        break;

    case type_tags::Reference:
        std::get<type_tags::Reference>(*this)->as_array();
        update();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Array>(self());
}

const EntryArray& Entry::as_array() const
{
    if (index() != type_tags::Array)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Array>(self());
}

//==========================================================================================

EntryObject::EntryObject(Entry* parent) : m_self_(parent) {}

EntryObject::~EntryObject() {}

//==========================================================================================

Cursor<Entry> EntryArray::children() { return Cursor<Entry>(m_container_.begin(), m_container_.end()); }

Cursor<const Entry> EntryArray::children() const
{
    return Cursor<const Entry>(m_container_.cbegin(), m_container_.cend());
}
void EntryArray::resize(std::size_t num) { m_container_.resize(num, Entry(parent())); }

void EntryArray::clear() { m_container_.clear(); }

Entry& EntryArray::push_back()
{
    m_container_.push_back(Entry(parent()));
    return m_container_.back();
}

Entry EntryArray::pop_back()
{
    Entry res;
    m_container_.back().swap(res);
    m_container_.pop_back();
    return std::move(res);
}

//==========================================================================================
template <>
size_t EntryObjectDefault::size() const { return m_container_.size(); }

template <>
void EntryObjectDefault::clear() { m_container_.clear(); }

template <>
Entry EntryObjectDefault::insert(const std::string& name) { return Entry(parent(), XPath(name)); }

template <>
Entry EntryObjectDefault::insert(const XPath& path) { return Entry(parent(), path); }
// {
//     Entry p(self());
//     for (auto it = path.begin(); it != path.end(); ++it)
//     {
//         switch (it->index())
//         {
//         case XPath::type_tags::Key:
//             p.as_object().insert(std::get<XPath::type_tags::Key>(*it)).swap(p);
//             break;
//         case XPath::type_tags::Index:
//             p.as_array().get(std::get<XPath::type_tags::Index>(*it)).swap(p);
//             break;
//         default:
//             NOT_IMPLEMENTED;
//             break;
//         }
//     }
//     return std::move(p);
// }

template <>
const Entry
EntryObjectDefault::get(const std::string& path) const { return m_container_.at(path); }

template <>
const Entry
EntryObjectDefault::get(const XPath& path) const
{

    Entry p = *self();

    for (auto it = path.begin(); it != path.end(); ++it)
    {
        switch (it->index())
        {
        case XPath::type_tags::Key:
            p = p.as_object().get(std::get<XPath::type_tags::Key>(*it));
            break;
        case XPath::type_tags::Index:
            p = p.as_array().get(std::get<XPath::type_tags::Index>(*it));
            break;
        default:
            NOT_IMPLEMENTED;
            break;
        }
    }
    return std::move(p);
}

template <>
void EntryObjectDefault::erase(const std::string& path) { m_container_.erase(m_container_.find(path)); }

template <>
void EntryObjectDefault::erase(const XPath& path) { NOT_IMPLEMENTED; }

//--------------------------------------------------------------------------------------------------------------------------------------------
template <>
Cursor<Entry>
EntryObjectDefault::select(const XPath& path)
{
    NOT_IMPLEMENTED;
    return Cursor<Entry>{};
}

template <>
Cursor<const Entry>
EntryObjectDefault::select(const XPath& path) const
{
    NOT_IMPLEMENTED;
    return Cursor<const Entry>{};
}

template <>
Cursor<Entry>
EntryObjectDefault::children()
{
    return Cursor<Entry>(m_container_.begin(), m_container_.end(), [](auto&& item) -> Entry& { return *item.second; });
}

template <>
Cursor<const Entry>
EntryObjectDefault::children() const
{
    return Cursor<const Entry>(m_container_.cbegin(), m_container_.cend(), [](auto&& item) -> const Entry& { return *item.second; });
}

template <>
Cursor<std::pair<const std::string, Entry>>
EntryObjectDefault::kv_items()
{
    return Cursor<std::pair<const std::string, Entry>>(m_container_.begin(), m_container_.end());
};

template <>
Cursor<const std::pair<const std::string, Entry>>
EntryObjectDefault::kv_items() const
{
    return Cursor<const std::pair<const std::string, Entry>>(m_container_.cbegin(), m_container_.cend());
};

//--------------------------------------------------------------------------------

template <>
size_t EntryArrayDefault::size() const { return m_container_.size(); }

template <>
void EntryArrayDefault::resize(std::size_t num)
{

    auto s = m_container_.size();
    m_container_.resize(num);
    for (int i = s; i < num; ++i)
    {
        if (m_container_[i] == nullptr)
        {
            m_container_[i].reset(new Entry(self()));
        }
    }
}

template <>
void EntryArrayDefault::clear() { m_container_.clear(); }

template <>
Cursor<Entry>
EntryArrayDefault::children()
{
    return Cursor<Entry>(m_container_.begin(), m_container_.end(),
                         [](const Entry& item) -> Entry& { return *item; });
}

template <>
Cursor<const Entry>
EntryArrayDefault::children() const
{
    return Cursor<const Entry>(m_container_.begin(), m_container_.end(),
                               [](const Entry& item) -> const Entry& { return *item; });
}

//--------------------------------------------------------------------------------------
template <>
Entry EntryArrayDefault::push_back()
{
    auto& p = m_container_.emplace_back();
    if (p == nullptr)
    {
        p.reset(new Entry(self()));
    }
    return p;
}

template <>
void EntryArrayDefault::pop_back() { m_container_.pop_back(); }

template <>
const Entry
EntryArrayDefault::get(int idx) const { return m_container_.at(idx); }

template <>
Entry EntryArrayDefault::get(int idx) { return m_container_.at(idx); }

} // namespace sp::db
namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    std::visit(sp::traits::overloaded{
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Array, sp::db::Entry::base_type>& ele) {
                       os << "[";
                       for (auto it = ele->children(); !it.done(); it.next())
                       {
                           os << std::endl
                              << std::setw(indent * tab) << " ";
                           fancy_print(os, *it, indent + 1, tab);
                           os << ",";
                       }
                       os << std::endl
                          << std::setw(indent * tab)
                          << "]";
                   },
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Object, sp::db::Entry::base_type>& ele) {
                       os << "{";
                       for (auto it = ele->kv_items(); !it.done(); it.next())
                       {
                           os << std::endl
                              << std::setw(indent * tab) << " "
                              << "\"" << it->first << "\" : ";
                           fancy_print(os, *(it->second), indent + 1, tab);
                           os << ",";
                       }
                       os << std::endl
                          << std::setw(indent * tab)
                          << "}";
                   },
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Empty, sp::db::Entry::base_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },
                   [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); } //
               },
               dynamic_cast<const sp::db::Entry::base_type&>(entry));

    // if (entry.type() == Entry::NodeType::Element)
    // {
    //     os << to_string(entry.get_element());
    // }
    // else if (entry.type() == Entry::NodeType::Array)
    // else if (entry.type() == Entry::NodeType::Object)
    // {
    //
    // }
    return os;
}
} // namespace sp::utility

namespace sp::db
{
std::ostream& operator<<(std::ostream& os, Entry const& entry) { return sp::utility::fancy_print(os, entry, 0); }
} // namespace sp::db