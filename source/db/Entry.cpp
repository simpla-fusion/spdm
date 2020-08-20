#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{

//-------------------------------------------------------------------------------

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

Entry EntryObject::insert(const Path& path)
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryObject::insert(const Path& path, const Entry& v) { insert(path).set_value(v); }

Entry EntryObject::find(const Path& path) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryObject::remove(const Path& path) { NOT_IMPLEMENTED; }

void EntryObject::update(const EntryObject& patch) { NOT_IMPLEMENTED; }

void EntryObject::merge(const EntryObject&) { NOT_IMPLEMENTED; }

void EntryObject::patch(const EntryObject&) { NOT_IMPLEMENTED; }

void EntryObject::emplace(const std::string& key, Entry&&) { NOT_IMPLEMENTED; }

//----------------------------------------------------------------------------------------------------

class EntryObjectDefault : public EntryObject
{
private:
    std::map<std::string, Entry> m_container_;
    static bool is_registered;
    static int associated_num;

public:
    typedef EntryObjectDefault this_type;
    typedef Entry::value_type_tags value_type_tags;

    EntryObjectDefault() = default;

    EntryObjectDefault(const this_type& other) : m_container_(other.m_container_) {}

    EntryObjectDefault(this_type&& other) : m_container_(std::move(other.m_container_)) {}

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

    // Cursor<Entry> select(const Path& path) override;

    // Cursor<const Entry> select(const Path& path) const override;

    void insert(const Path& path, const Entry&) override;

    Entry find(const Path& path) const override;

    void remove(const Path& path) override;

    void update(const EntryObject& patch) override;
};

void EntryObjectDefault::insert(const Path& path, const Entry& v)
{

    auto it = path.begin();
    auto ie = path.end();
    if (it == ie || it->index() != Path::segment_tags::Key)
    {
        RUNTIME_ERROR << "Illegal path";
    }
    else
    {

        auto current = m_container_.try_emplace(std::get<Path::segment_tags::Key>(*it)).first;
        ++it;
        // current->second.insert(Path(it, ie)).update(v);
    }
}

//---------------------------------------------------------------------------------------------------

Entry EntryObjectDefault::find(const Path& path) const
{

    Entry p;

    // for (auto it = path.begin(); it != path.end(); ++it)
    // {
    //     switch (it->index())
    //     {
    //     case Path::value_type_tags::Key:
    //         p = p.as_object().get(std::get<Path::value_type_tags::Key>(*it));
    //         break;
    //     case Path::value_type_tags::Index:
    //         p = p.as_array().get(std::get<Path::value_type_tags::Index>(*it));
    //         break;
    //     default:
    //         NOT_IMPLEMENTED;
    //         break;
    //     }
    // }
    return std::move(p);
}

void EntryObjectDefault::remove(const Path& path) { m_container_.erase(m_container_.find(path.str())); }

void EntryObjectDefault::update(const EntryObject& patch) { NOT_IMPLEMENTED; }

//==========================================================================================
// EntryArray

void EntryArray::resize(std::size_t num) { m_container_.resize(num); }

size_t EntryArray::size() const { return m_container_.size(); }

void EntryArray::clear() { m_container_.clear(); }

Entry EntryArray::at(int idx) { return Entry(m_container_.at(idx)); }

const Entry EntryArray::at(int idx) const { return Entry(m_container_.at(idx)); }

Entry EntryArray::slice(int start, int stop, int step)
{
    NOT_IMPLEMENTED;
    return Entry{};
}

const Entry EntryArray::slice(int start, int stop, int step) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

Entry EntryArray::insert(const Path& path)
{
    NOT_IMPLEMENTED;
    return Entry{};
}

Entry EntryArray::find(const Path& path) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryArray::push_back(const Entry& v)
{
    NOT_IMPLEMENTED;
}

Entry EntryArray::push_back()
{
    NOT_IMPLEMENTED;
    return Entry{};
}
void EntryArray::emplace_back(Entry&& v)
{
    NOT_IMPLEMENTED;
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

//===========================================================================================================
// Entry
//-----------------------------------------------------------------------------------------------------------
size_t Entry::size() const
{
    size_t res = 0;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { res = obj_p->size(); },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { res = array_p->size(); },
            [&](auto&&) { res = 0; }},
        dynamic_cast<const base_type&>(*this));
    return res;
}

Cursor<Entry> Entry::Entry::children()
{
    Cursor<Entry> res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { obj_p->children().swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->children().swap(res); },
            [&](auto&&) {}},
        dynamic_cast<base_type&>(*this));
    return std::move(res);
}

Cursor<const Entry> Entry::Entry::children() const
{
    Cursor<const Entry> res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { std::dynamic_pointer_cast<const Entry>(obj_p)->children().swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { std::dynamic_pointer_cast<const Entry>(array_p)->children().swap(res); },
            [&](auto&&) {}},
        dynamic_cast<const base_type&>(*this));
    return std::move(res);
}
//-----------------------------------------------------------------------------------------------------------

void Entry::set_value(Entry&& v)
{

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { base_type::operator=(v); }},
        dynamic_cast<base_type&>(*this));
}

const Entry Entry::get_value() const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { res = v; }},
        dynamic_cast<const base_type&>(*this));
    return std::move(res);
}

EntryObject& Entry::as_object()
{
    EntryObject* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { res = obj_p.get(); },
            [&](std::variant_alternative_t<value_type_tags::Null, base_type>&) {
                auto p = std::make_shared<EntryObjectDefault>();
                base_type::emplace<value_type_tags::Object>(p);
                res = p.get();
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        dynamic_cast<base_type&>(*this));

    return *res;
}

const EntryObject& Entry::as_object() const
{
    EntryObject* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { res = obj_p.get(); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        dynamic_cast<const base_type&>(*this));

    return *res;
}

Entry Entry::insert(const Path& path) { return as_object().insert(path); }

EntryArray& Entry::as_array()
{
    EntryArray* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { res = array_p.get(); },
            [&](std::variant_alternative_t<value_type_tags::Null, base_type>&) {
                auto p = std::make_shared<EntryArray>();
                base_type::emplace<value_type_tags::Array>(p);
                res = p.get();
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        dynamic_cast<base_type&>(*this));

    return *res;
}

const EntryArray& Entry::as_array() const
{
    EntryArray* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& obj_p) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { res = array_p.get(); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Array!"; }},
        dynamic_cast<const base_type&>(*this));

    return *res;
}

void Entry::resize(std::size_t num) { as_array().resize(num); }

Entry Entry::pop_back() { return as_array().pop_back(); }

Entry Entry::operator[](const std::string& path)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(path)).swap(res);
            },
            [&](std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { obj_p->insert(path).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<base_type&>(*this));
    return std::move(res);
}

const Entry Entry::operator[](const std::string& path) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(path)).swap(res);
            },
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { obj_p->find(path).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const base_type&>(*this));
    return std::move(res);
}

Entry Entry::operator[](int idx)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(idx)).swap(res);
            },
            [&](std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->at(idx).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<base_type&>(*this));
    return std::move(res);
}
const Entry Entry::operator[](int idx) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(idx)).swap(res);
            },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->at(idx).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const base_type&>(*this));
    return std::move(res);
}

Entry Entry::slice(int start, int stop, int step)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(std::make_tuple(start, stop, step))).swap(res);
            },
            [&](std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->slice(start, stop, step).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<base_type&>(*this));
    return std::move(res);
}
const Entry Entry::slice(int start, int stop, int step) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(std::make_tuple(start, stop, step))).swap(res);
            },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->slice(start, stop, step).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const base_type&>(*this));
    return std::move(res);
}

Entry Entry::operator[](const Path& path)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(path)).swap(res);
            },
            [&](std::variant_alternative_t<value_type_tags::Object, base_type>& object_p) { object_p->insert(path).swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->insert(path).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<base_type&>(*this));
    return std::move(res);
}

const Entry Entry::operator[](const Path& path) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, base_type>& ref) {
                Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(path)).swap(res);
            },
            [&](const std::variant_alternative_t<value_type_tags::Object, base_type>& obj_p) { obj_p->find(path).swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, base_type>& array_p) { array_p->find(path).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const base_type&>(*this));
    return std::move(res);
}

} // namespace sp::db
namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    std::visit(sp::traits::overloaded{
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Array, sp::db::Entry::base_type>& array_p) {
                       os << "[";
                       for (auto it = array_p->children(); !it.done(); it.next())
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
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Object, sp::db::Entry::base_type>& obj_p) {
                       os << "{";
                       for (auto it = obj_p->children(); !it.done(); it.next())
                       {
                           os << std::endl
                              << std::setw(indent * tab) << " ";
                           fancy_print(os, *it, indent + 1, tab);
                           os << ",";
                       }
                       os << std::endl
                          << std::setw(indent * tab)
                          << "}";
                   },
                   //    [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Item, sp::db::Entry::base_type>& item) {
                   //        os << item << ":";
                   //        fancy_print(os, item.second, indent + 1, tab);
                   //    },
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Reference, sp::db::Entry::base_type>& ref) {
                       os << "<" << ref.second.str() << ">";
                   },                                                                                                                                                                 //
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Block, sp::db::Entry::base_type>& ele) { fancy_print(os, "<DATA BLOCK>", indent + 1, tab); }, //
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Null, sp::db::Entry::base_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },         //
                   [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); }                                                                                                         //
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