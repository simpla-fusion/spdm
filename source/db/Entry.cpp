#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{
void insert(entry_value_type& value, const Path& path, entry_value_type&& v)
{
}
//----------------------------------------------------------------------------------------------------
EntryContainer::~EntryContainer() {}

void EntryContainer::insert(const Path& path, entry_value_type&& v)
{
    try_insert(path).set_value(std::forward<entry_value_type>(v));
}

Entry EntryContainer::try_insert(const Path& path, entry_value_type&& v)
{
    Entry res;

    // for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    // {
    //     std::visit(
    //         sp::traits::overloaded{
    //             [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& obj_p) { current = &obj_p->try_insert_node(*it); },
    //             [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { current = &array_p->try_insert_node(*it); },
    //             [&](std::variant_alternative_t<entry_value_type_tags::Reference, entry_value_type>& ref) { NOT_IMPLEMENTED; },
    //             [&](auto&&) { NOT_IMPLEMENTED; }},
    //         *current);
    // }
    return std::move(res);
}

void EntryContainer::remove(const Path& path)
{
    // find(path.prefix()).remove(path.last());
}

const Entry EntryContainer::find(const Path& path) const
{
    Entry res;
    return std::move(res);
}

EntryObject::~EntryObject() {}

void EntryObject::insert(const Path::PathSegment& key, entry_value_type&&) {}

const Entry EntryObject::find(const Path::PathSegment& key) const { return Entry{}; }

void EntryObject::remove(const Path::PathSegment& path) {}
//----------------------------------------------------------------------------------------------------

class EntryObjectDefault : public EntryObject
{
private:
    std::map<std::string, entry_value_type> m_container_;

public:
    typedef EntryObjectDefault this_type;
    typedef Entry::value_type_tags value_type_tags;

    using EntryObject::find;
    using EntryObject::insert;
    using EntryObject::remove;

    EntryObjectDefault() {};

    EntryObjectDefault(const this_type& other) : m_container_(other.m_container_) {}

    EntryObjectDefault(this_type&& other) : m_container_(std::move(other.m_container_)) {}

    virtual ~EntryObjectDefault() {}

    std::unique_ptr<EntryContainer> copy() const override { return std::unique_ptr<EntryContainer>(new EntryObjectDefault(*this)); }

    size_t size() const override { return m_container_.size(); }

    void clear() override { return m_container_.clear(); }

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    const Entry find(const Path::PathSegment& key) const override;

    void insert(const Path::PathSegment& key, entry_value_type&&) override;

    void remove(const Path::PathSegment& path) override;

    void merge(const EntryObject&) override;

    void patch(const EntryObject&) override;

    void update(const EntryObject& patch) override;
};
const Entry EntryObjectDefault::find(const Path::PathSegment& key) const { return Entry{}; }

void EntryObjectDefault::insert(const Path::PathSegment& key, entry_value_type&&){};

void EntryObjectDefault::remove(const Path::PathSegment& path) { m_container_.erase(m_container_.find(std::get<std::string>(path))); }

void EntryObjectDefault::merge(const EntryObject&) { NOT_IMPLEMENTED; };

void EntryObjectDefault::update(const EntryObject& patch) { NOT_IMPLEMENTED; }

void EntryObjectDefault::patch(const EntryObject& patch) { NOT_IMPLEMENTED; }

//==========================================================================================
// EntryArray

Cursor<Entry> EntryArray::children() { return Cursor<Entry>(); /*(m_container_.begin(), m_container_.end());*/ }

Cursor<const Entry> EntryArray::children() const { return Cursor<const Entry>(); /*(m_container_.cbegin(), m_container_.cend());*/ }

void EntryArray::insert(const Path::PathSegment& path, entry_value_type&& v)
{
    NOT_IMPLEMENTED;
}

const Entry EntryArray::find(const Path::PathSegment& path) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryArray::remove(const Path::PathSegment& path)
{
    NOT_IMPLEMENTED;
}

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
    res.set_value(m_container_.back());
    m_container_.pop_back();
    return std::move(res);
}

//===========================================================================================================
// Entry
//-----------------------------------------------------------------------------------------------------------

Entry::Entry() {}

void Entry::swap(Entry& other)
{
    std::swap(m_value_, other.m_value_);
}

size_t Entry::size() const
{
    size_t res = 0;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) { NOT_IMPLEMENTED; },
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { res = obj_p->size(); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p->size(); },
            [&](auto&&) { res = 0; }},
        dynamic_cast<const value_type&>(m_value_));
    return res;
}

Cursor<Entry> Entry::Entry::children()
{
    Cursor<Entry> res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { obj_p->children().swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
            [&](auto&&) {}},
        dynamic_cast<value_type&>(m_value_));
    return std::move(res);
}

Cursor<const Entry> Entry::Entry::children() const
{
    Cursor<const Entry> res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { std::dynamic_pointer_cast<const Entry>(obj_p)->children().swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { std::dynamic_pointer_cast<const Entry>(array_p)->children().swap(res); },
            [&](auto&&) {}},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

Entry Entry::insert(const Path& path)
{
    Entry res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) { Entry(std::in_place_index_t<value_type_tags::Reference>(), ref.first, ref.second.join(path)).swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { Entry(std::in_place_index_t<value_type_tags::Reference>(), obj_p, path).swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { Entry(std::in_place_index_t<value_type_tags::Reference>(), array_p, path).swap(res); },
            [&](auto&& v) { NOT_IMPLEMENTED; }},
        m_value_);
    return std::move(res);
}

Entry Entry::find(const Path& path) const
{
    Entry res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) { Entry(ref.first->find(ref.second.join(path))).swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { Entry(obj_p->find(path)).swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { Entry(array_p->find(path)).swap(res); },
            [&](auto&& v) { if(path.empty()){res=v;} }},
        m_value_);
    return std::move(res);
}

//-----------------------------------------------------------------------------------------------------------
Entry::value_type& Entry::value() { return m_value_; }

const Entry::value_type& Entry::value() const { return m_value_; }

void Entry::set_value(value_type&& v)
{
    if (m_value_.index() == value_type_tags::Reference)
    {
        auto ref = std::get<value_type_tags::Reference>(m_value_);
        ref.first->insert(ref.second, std::forward<value_type>(v));
    }
    else
    {
        value_type(v).swap(m_value_);
    }
}

void Entry::set_value(const value_type& v) { set_value(value_type(v)); }

Entry::value_type Entry::get_value()
{
    value_type res;

    if (m_value_.index() == value_type_tags::Reference)
    {
        NOT_IMPLEMENTED;
        // m_parent_.lock()->find(std::get<value_type_tags::Reference>(m_value_)).swap(res);
    }
    else
    {
        value_type(m_value_).swap(res);
    }
    return std::move(res);
}

Entry::value_type Entry::get_value() const
{
    value_type res;

    if (m_value_.index() == value_type_tags::Reference)
    {
        NOT_IMPLEMENTED;
        // m_parent_.lock()->find(std::get<value_type_tags::Reference>(m_value_)).swap(res);
    }
    else
    {
        value_type(m_value_).swap(res);
    }
    return std::move(res);
}

EntryObject& Entry::as_object()
{
    EntryObject* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) { res = &ref.first->try_insert(ref.second).as_object(); },
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { res = obj_p.get(); },
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>&) {
                auto p = std::make_shared<EntryObjectDefault>();
                m_value_.emplace<value_type_tags::Object>(p);
                res = p.get();
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        m_value_);

    return *res;
}

const EntryObject& Entry::as_object() const
{
    const EntryObject* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                //  res = &m_parent_.lock()->insert(ref.second).as_object();
                NOT_IMPLEMENTED;
            },
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { res = obj_p.get(); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        m_value_);

    return *res;
}

EntryArray& Entry::as_array()
{
    EntryArray* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) { res = &ref.first->try_insert(ref.second).as_array(); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p.get(); },
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>&) {
                auto p = std::make_shared<EntryArray>();
                m_value_.emplace<value_type_tags::Array>(p);
                res = p.get();
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        m_value_);

    return *res;
}

const EntryArray& Entry::as_array() const
{
    const EntryArray* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) { res = &ref.first->try_insert(ref.second).as_array(); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p.get(); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Array!"; }},
        m_value_);

    return *res;
}

void Entry::resize(std::size_t num) { as_array().resize(num); }

Entry Entry::pop_back() { return as_array().pop_back(); }

Entry Entry::operator[](const std::string& path)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(path));
            },
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) {
                res.m_value_.emplace<value_type_tags::Reference>(obj_p, Path(path));
            },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<value_type&>(m_value_));
    return std::move(res);
}

const Entry Entry::operator[](const std::string& path) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(path));
            },
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { Entry(obj_p->find(path)).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

Entry Entry::operator[](int idx)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(idx));
            },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->at(idx).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<value_type&>(m_value_));
    return std::move(res);
}

const Entry Entry::operator[](int idx) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(idx));
            },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->at(idx).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

Entry Entry::slice(int start, int stop, int step)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(std::make_tuple(start, stop, step)));
            },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->slice(start, stop, step).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<value_type&>(m_value_));
    return std::move(res);
}

const Entry Entry::slice(int start, int stop, int step) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(std::make_tuple(start, stop, step)));
            },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->slice(start, stop, step).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

Entry Entry::operator[](const Path& path)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(path));
            },
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->try_insert(path).swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->try_insert(path).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<value_type&>(m_value_));
    return std::move(res);
}

const Entry Entry::operator[](const Path& path) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Reference, value_type>& ref) {
                res.m_value_.emplace<value_type_tags::Reference>(ref.first, ref.second.join(path));
            },
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& obj_p) { Entry(obj_p->find(path)).swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { Entry(array_p->find(path)).swap(res); },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

} // namespace sp::db
namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    std::visit(sp::traits::overloaded{
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Array, sp::db::Entry::value_type>& array_p) {
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
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Object, sp::db::Entry::value_type>& obj_p) {
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
                   //    [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Item, sp::db::Entry::value_type>& item) {
                   //        os << item << ":";
                   //        fancy_print(os, item.second, indent + 1, tab);
                   //    },
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Reference, sp::db::Entry::value_type>& ref) {
                       os << "<" << ref.second.str() << ">";
                   },                                                                                                                                                                    //
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Block, sp::db::Entry::value_type>& blk_p) { fancy_print(os, "<DATA BLOCK>", indent + 1, tab); }, //
                   [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Null, sp::db::Entry::value_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },           //
                   [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); }                                                                                                            //
               },
               dynamic_cast<const sp::db::Entry::value_type&>(entry.value()));

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