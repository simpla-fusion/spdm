#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{
EntryObject::EntryObject(Entry* holder) : m_holder_(holder) {}

//-------------------------------------------------------------------------------
Entry* EntryObject::holder() const { return m_holder_; }

std::unique_ptr<EntryObject> EntryObject::copy() const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

size_t EntryObject::size() const { NOT_IMPLEMENTED; }

void EntryObject::clear() { NOT_IMPLEMENTED; }

Cursor<Entry> EntryObject::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Entry>{};
}

Cursor<const Entry> EntryObject::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Entry>{};
}

Cursor<std::pair<const std::string, Entry>> EntryObject::kv_items()
{
    NOT_IMPLEMENTED;
    return Cursor<std::pair<const std::string, Entry>>{};
}

Cursor<const std::pair<const std::string, Entry>> EntryObject::kv_items() const
{
    NOT_IMPLEMENTED;
    return Cursor<const std::pair<const std::string, Entry>>{};
}

// Cursor<Entry> EntryObject::select(const XPath& path)
// {
//     NOT_IMPLEMENTED;
//     return Cursor<Entry>{};
// }

// Cursor<const Entry> EntryObject::select(const XPath& path) const
// {
//     NOT_IMPLEMENTED;
//     return Cursor<const Entry>{};
// }

void EntryObject::insert(const XPath& path, const Entry&) { NOT_IMPLEMENTED; }

Entry EntryObject::query(const XPath& path) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryObject::remove(const XPath& path) { NOT_IMPLEMENTED; }

void EntryObject::update(const EntryObject& patch) { NOT_IMPLEMENTED; }

Entry EntryObject::operator[](const XPath& path) { return Entry(m_holder_, path); }

const Entry EntryObject::operator[](const XPath& path) const { return Entry(const_cast<Entry*>(m_holder_), path); }

//----------------------------------------------------------------------------------------------------
class EntryObjectDefault : public EntryObject
{
private:
    std::map<std::string, Entry> m_container_;
    static bool is_registered;
    static int associated_num;

public:
    typedef EntryObjectDefault this_type;
    typedef Entry::type_tags type_tags;

    EntryObjectDefault(Entry* self) : EntryObject(self) {}

    EntryObjectDefault(const this_type& other) : EntryObject(nullptr), m_container_(other.m_container_) {}

    EntryObjectDefault(this_type&& other) : EntryObject(nullptr), m_container_(std::move(other.m_container_)) {}

    ~EntryObjectDefault() = default;

    std::unique_ptr<EntryObject> copy() const override { return std::unique_ptr<EntryObject>(new this_type(*this)); }

    //----------------------------------------------------------------------------------------------------------

    size_t size() const override { return m_container_.size(); }

    void clear() override { return m_container_.clear(); }

    Cursor<Entry> children() override
    {
        return Cursor<Entry>(m_container_.begin(), m_container_.end(), [](auto&& item) -> Entry& { return item.second; });
    }

    Cursor<const Entry> children() const override
    {
        return Cursor<const Entry>(m_container_.cbegin(), m_container_.cend(), [](auto&& item) -> const Entry& { return item.second; });
    }

    Cursor<std::pair<const std::string, Entry>> kv_items() override
    {
        return Cursor<std::pair<const std::string, Entry>>(m_container_.begin(), m_container_.end());
    }

    Cursor<const std::pair<const std::string, Entry>> kv_items() const override
    {
        return Cursor<const std::pair<const std::string, Entry>>(m_container_.cbegin(), m_container_.cend());
    }

    // Cursor<Entry> select(const XPath& path) override;

    // Cursor<const Entry> select(const XPath& path) const override;

    void insert(const XPath& path, const Entry&) override;

    Entry query(const XPath& path) const override;

    void remove(const XPath& path) override;

    void update(const EntryObject& patch) override;
};

void EntryObjectDefault::insert(const XPath& path, const Entry&)
{
}
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
//---------------------------------------------------------------------------------------------------

Entry EntryObjectDefault::query(const XPath& path) const
{

    Entry p = *holder();

    // for (auto it = path.begin(); it != path.end(); ++it)
    // {
    //     switch (it->index())
    //     {
    //     case XPath::type_tags::Key:
    //         p = p.as_object().get(std::get<XPath::type_tags::Key>(*it));
    //         break;
    //     case XPath::type_tags::Index:
    //         p = p.as_array().get(std::get<XPath::type_tags::Index>(*it));
    //         break;
    //     default:
    //         NOT_IMPLEMENTED;
    //         break;
    //     }
    // }
    return std::move(p);
}

void EntryObjectDefault::remove(const XPath& path) { m_container_.erase(m_container_.find(path)); }

// Cursor<Entry>
// EntryObjectDefault::select(const XPath& path)
// {
//     NOT_IMPLEMENTED;
//     return Cursor<Entry>{};
// }

// Cursor<const Entry>
// EntryObjectDefault::select(const XPath& path) const
// {
//     NOT_IMPLEMENTED;
//     return Cursor<const Entry>{};
// }

//
// Cursor<std::pair<const std::string, Entry>>
// EntryObjectDefault::kv_items()
// {
//     return Cursor<std::pair<const std::string, Entry>>(m_container_.begin(), m_container_.end());
// };

//
// Cursor<const std::pair<const std::string, Entry>>
// EntryObjectDefault::kv_items() const
// {
//     return Cursor<const std::pair<const std::string, Entry>>(m_container_.cbegin(), m_container_.cend());
// };

//-----------------------------------------------------------------------------------------------------------

Entry::Entry(Entry* parent) : m_parent_(parent) {}

Entry::~Entry() {}

Entry::Entry(const Entry& other) : base_type(other), m_parent_(other.m_parent_) {}

Entry::Entry(Entry&& other) : base_type(std::move(other)), m_parent_(other.m_parent_) {}

void Entry::swap(Entry& other) { base_type::swap(other); }

Entry Entry::insert(const XPath& path)
{
    /** NOTE: lazy access is not thread safe; */

    if (index() == type_tags::Reference)
    {
        return Entry(parent(), std::get<type_tags::Reference>(*this).join(path));
    }
    else
    {
        return Entry(this, path);
    }
}

void Entry::insert(const XPath& path, const Entry& v)
{
    std::visit(sp::traits::overloaded{
                   [&](std::nullptr_t) { as_object().insert(path, v); },
                   [&](std::variant_alternative_t<type_tags::Reference, base_type>& p) { Entry(parent(), p.join(path)).update(v); },
                   [&](std::variant_alternative_t<type_tags::Object, base_type>& obj) { obj->insert(path, v); },
                   [&](std::variant_alternative_t<type_tags::Array, base_type>& obj) { obj.insert(path, v); },
                   [&](auto&& ele) { RUNTIME_ERROR << "Try to add child to leaf node"; }},
               dynamic_cast<base_type&>(*this));
}

Entry Entry::query(const XPath& path) const
{
    Entry res;

    std::visit(sp::traits::overloaded{
                   [&](std::nullptr_t) {},
                   [&](const std::variant_alternative_t<type_tags::Reference, base_type>& ele) {
                       if (is_root())
                       {
                           NOT_IMPLEMENTED;
                       }
                       else
                       {
                           parent()->query(std::get<type_tags::Reference>(*this).join(path)).swap(res);
                       }
                   },
                   [&](const std::variant_alternative_t<type_tags::Object, base_type>& obj) { obj->query(path).swap(res); },
                   [&](const std::variant_alternative_t<type_tags::Array, base_type>& obj) { obj.query(path).swap(res); },
                   [&](auto&& ele) {
                       if (path.empty())
                       {
                           res = ele;
                       }
                   }},
               dynamic_cast<const base_type&>(*this));

    return std::move(res);
}

void Entry::remove(const XPath&) {}

void Entry::update(const Entry& v)
{
    if (index() != type_tags::Reference)
    {
        Entry(v).swap(*this);
    }
    else if (parent() != nullptr)
    {
        parent()->insert(std::get<type_tags::Reference>(*this), v);
    }
    else
    {
        RUNTIME_ERROR << "Empty referecne!" << std::get<type_tags::Reference>(*this).str();
    }
}

EntryObject& Entry::as_object()
{
    switch (base_type::index())
    {
    case type_tags::Null:
        emplace<type_tags::Object>(std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryObjectDefault>(this)));
        break;
    case type_tags::Object:
        break;
    case type_tags::Reference:
        // std::get<type_tags::Reference>(*this)->as_object();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return *std::get<type_tags::Object>(*this);
}

const EntryObject& Entry::as_object() const
{
    if (type() != type_tags::Object)
    {
        throw std::runtime_error("illegal type");
    }
    return *std::get<type_tags::Object>(*this);
}

EntryArray& Entry::as_array()
{
    switch (base_type::index())
    {
    case type_tags::Null:
        emplace<type_tags::Array>(EntryArray(this));
        break;
    case type_tags::Array:
        break;

    case type_tags::Reference:
        // std::get<type_tags::Reference>(*this)->as_array();
        // update();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Array>(*this);
}

const EntryArray& Entry::as_array() const
{
    if (index() != type_tags::Array)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Array>(*this);
}

Entry Entry::operator[](const XPath& path) const
{
    return Entry{const_cast<Entry*>(this), path};
}
//==========================================================================================

void EntryArray::resize(std::size_t num) { m_container_.resize(num, Entry(holder())); }

size_t EntryArray::size() const { return m_container_.size(); }

void EntryArray::clear() { m_container_.clear(); }

Entry& EntryArray::push_back()
{
    m_container_.push_back(Entry(holder()));
    return m_container_.back();
}

Entry EntryArray::pop_back()
{
    Entry res;
    m_container_.back().swap(res);
    m_container_.pop_back();
    return std::move(res);
}

Cursor<Entry> EntryArray::children() { return Cursor<Entry>(m_container_.begin(), m_container_.end()); }

Cursor<const Entry> EntryArray::children() const { return Cursor<const Entry>(m_container_.cbegin(), m_container_.cend()); }

} // namespace sp::db
namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    std::visit(sp::traits::overloaded{
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Array, sp::db::Entry::base_type>& ele) {
                       os << "[";
                       for (auto it = ele.children(); !it.done(); it.next())
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
                           fancy_print(os, it->second, indent + 1, tab);
                           os << ",";
                       }
                       os << std::endl
                          << std::setw(indent * tab)
                          << "}";
                   },
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Reference, sp::db::Entry::base_type>& ele) { fancy_print(os, ele.str(), indent + 1, tab); },  //
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Block, sp::db::Entry::base_type>& ele) { fancy_print(os, "<DATA BLOCK>", indent + 1, tab); }, //
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Null, sp::db::Entry::base_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },         //
                   [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); }                                                                                                   //
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