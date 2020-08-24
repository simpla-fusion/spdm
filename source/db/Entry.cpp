#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{

//----------------------------------------------------------------------------------------------------

void EntryContainer::insert(const Path& path, const entry_value_type& v)
{
    VERBOSE << path.str();

    EntryContainer* current = this;
    auto it = path.begin();
    auto ie = path.end();
    Path::Segment seg;
    do
    {
        seg = *it;
        ++it;
        if (it != ie)
        {
            current = current->sub_container(seg);
        }
    } while ((it != ie));

    current->insert(seg, v);
}

void EntryContainer::remove(const Path& path)
{
    // find(path.prefix()).remove(path.last());
}

const entry_value_type EntryContainer::find(const Path& path) const
{
    entry_value_type res;
    return std::move(res);
}

const entry_value_type EntryContainer::at(const Path& path) const
{
    return entry_value_type{};
}

entry_value_type EntryContainer::at(const Path& path)
{
    return entry_value_type{};
}

//----------------------------------------------------------------------------------------------------
size_t EntryObject::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

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
EntryContainer* EntryObject::sub_container(const Path::Segment& key) { return nullptr; }

const EntryContainer* EntryObject::sub_container(const Path::Segment& key) const { return nullptr; }

void EntryObject::insert(const Path::Segment& key, const entry_value_type&) { NOT_IMPLEMENTED; }

const entry_value_type EntryObject::find(const Path::Segment& key) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

void EntryObject::remove(const Path::Segment& path) { NOT_IMPLEMENTED; }
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

    EntryObjectDefault() = default;

    EntryObjectDefault(const this_type& other) : m_container_(other.m_container_) {}

    EntryObjectDefault(this_type&& other) : m_container_(std::move(other.m_container_)) {}

    virtual ~EntryObjectDefault() = default;

    std::unique_ptr<EntryContainer> copy() const override { return std::unique_ptr<EntryContainer>(new EntryObjectDefault(*this)); }

    size_t size() const override { return m_container_.size(); }

    void clear() override { return m_container_.clear(); }

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    void for_each(std::function<void(const std::string&, entry_value_type&)> const&) override;

    void for_each(std::function<void(const std::string&, const entry_value_type&)> const&) const override;

    EntryContainer* sub_container(const Path::Segment& key) override;

    const EntryContainer* sub_container(const Path::Segment& key) const override;

    const entry_value_type find(const Path::Segment& key) const override;

    void insert(const Path::Segment& key, const entry_value_type&) override;

    void remove(const Path::Segment& path) override;

    void merge(const EntryObject&) override;

    void patch(const EntryObject&) override;

    void update(const EntryObject& patch) override;
};

Cursor<Entry> EntryObjectDefault::children()
{
    return Cursor<Entry>{};
}

Cursor<const Entry> EntryObjectDefault::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Entry>{};
}
void EntryObjectDefault::for_each(std::function<void(const std::string&, entry_value_type&)> const& visitor)
{

    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}

void EntryObjectDefault::for_each(std::function<void(const std::string&, const entry_value_type&)> const& visitor) const
{
    for (auto&& item : m_container_)
    {
        visitor(item.first, item.second);
    }
}
EntryContainer* EntryObjectDefault::sub_container(const Path::Segment& key)
{
    EntryContainer* res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::string& k) {
                auto p = m_container_.try_emplace(k);
                if (!p.second)
                {
                    p.first->second.emplace<value_type_tags::Object>(new EntryObjectDefault);
                }
                res = std::get<value_type_tags::Object>(p.first->second).get();
            },

            [&](auto&&) { RUNTIME_ERROR << "illegal type! "; }},
        key);
    return res;
}

const EntryContainer* EntryObjectDefault::sub_container(const Path::Segment& key) const
{
    const EntryContainer* res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::string& k) {
                auto p = m_container_.find(k);
                if (p != m_container_.end() && p->second.index() == value_type_tags::Object)
                {
                    res = std::get<value_type_tags::Object>(p->second).get();
                }
                res = std::get<value_type_tags::Object>(p->second).get();
            },
            [&](auto&&) { RUNTIME_ERROR << "illegal type! "; }},
        key);
    return res;
}

const entry_value_type EntryObjectDefault::find(const Path::Segment& key) const
{
    entry_value_type res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::string& k) { entry_value_type(m_container_.at(k)).swap(res); },
            [&](auto&&) { RUNTIME_ERROR << "illegal key type! "; }},
        key);
    return std::move(res);
}

void EntryObjectDefault::insert(const Path::Segment& key, const entry_value_type& v)
{

    std::visit(
        sp::traits::overloaded{
            [&](const std::string& k) { m_container_.emplace(k, v); },
            [&](auto&&) { RUNTIME_ERROR << "illegal key type! "; }},
        key);
};

void EntryObjectDefault::remove(const Path::Segment& path) { m_container_.erase(m_container_.find(std::get<std::string>(path))); }

void EntryObjectDefault::merge(const EntryObject&) { NOT_IMPLEMENTED; };

void EntryObjectDefault::update(const EntryObject& patch) { NOT_IMPLEMENTED; }

void EntryObjectDefault::patch(const EntryObject& patch) { NOT_IMPLEMENTED; }

//==========================================================================================
// EntryArray

void EntryArray::resize(std::size_t num) { m_container_.resize(num); }

size_t EntryArray::size() const { return m_container_.size(); }

void EntryArray::clear() { m_container_.clear(); }

Cursor<Entry> EntryArray::children() { return Cursor<Entry>(); /*(m_container_.begin(), m_container_.end());*/ }

Cursor<const Entry> EntryArray::children() const { return Cursor<const Entry>(); /*(m_container_.cbegin(), m_container_.cend());*/ }

void EntryArray::for_each(std::function<void(int, entry_value_type&)> const& visitor)
{

    NOT_IMPLEMENTED;
}

void EntryArray::for_each(std::function<void(int, const entry_value_type&)> const& visitor) const
{
    NOT_IMPLEMENTED;
}

EntryContainer* EntryArray::sub_container(const Path::Segment& key)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

const EntryContainer* EntryArray::sub_container(const Path::Segment& key) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

void EntryArray::insert(const Path::Segment& path, const entry_value_type& v)
{
    NOT_IMPLEMENTED;
}

const entry_value_type EntryArray::find(const Path::Segment& path) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

void EntryArray::remove(const Path::Segment& path)
{
    NOT_IMPLEMENTED;
}

entry_value_type& EntryArray::at(int idx) { return (m_container_.at(idx)); }

const entry_value_type& EntryArray::at(int idx) const { return (m_container_.at(idx)); }

entry_value_type EntryArray::slice(int start, int stop, int step)
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

const entry_value_type EntryArray::slice(int start, int stop, int step) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

entry_value_type EntryArray::push_back()
{
    return emplace_back(nullptr);
}

entry_value_type EntryArray::emplace_back(entry_value_type&& v)
{
    m_container_.emplace_back(v);
    return entry_value_type{};
}

entry_value_type EntryArray::pop_back()
{
    entry_value_type res(m_container_.back());
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
    std::swap(m_path_, other.m_path_);
}

size_t Entry::size() const
{
    size_t res = 0;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { res = object_p->size(); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p->size(); },
            [&](auto&&) { res = 0; }},
        get_value());
    return res;
}

Cursor<Entry> Entry::Entry::children()
{
    Cursor<Entry> res;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->children().swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
            [&](auto&&) {}},
        m_value_);
    return std::move(res);
}

Cursor<const Entry> Entry::Entry::children() const
{
    Cursor<const Entry> res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { std::dynamic_pointer_cast<const Entry>(object_p)->children().swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { std::dynamic_pointer_cast<const Entry>(array_p)->children().swap(res); },
            [&](auto&&) { NOT_IMPLEMENTED; }},
        get_value());
    return std::move(res);
}

//-----------------------------------------------------------------------------------------------------------

void Entry::set_value(const entry_value_type& v)
{
    if (!m_path_.empty())
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->insert(m_path_, v); },
                [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->insert(m_path_, v); },
                [&](std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
                [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
            m_value_);
    }
    else
    {
        entry_value_type(v).swap(m_value_);
    }
}

entry_value_type Entry::get_value() const
{
    if (!m_path_.empty())
    {
        entry_value_type res;
        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { res = object_p->find(m_path_); },
                [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p->find(m_path_); },
                [&](const std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
                [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
            m_value_);

        return std::move(res);
    }
    else
    {
        return entry_value_type(m_value_);
    }
}

EntryObject& Entry::as_object()
{
    EntryObject* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { res = object_p.get(); },
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>&) {
                auto p = std::make_shared<EntryObjectDefault>();
                m_value_.emplace<value_type_tags::Object>(p);
                res = p.get();
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        get_value());

    return *res;
}

const EntryObject& Entry::as_object() const
{
    const EntryObject* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { res = object_p.get(); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        get_value());

    return *res;
}

EntryArray& Entry::as_array()
{
    EntryArray* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p.get(); },
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>&) {
                auto p = std::make_shared<EntryArray>();
                m_value_.emplace<value_type_tags::Array>(p);
                res = p.get();
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        get_value());

    return *res;
}

const EntryArray& Entry::as_array() const
{
    const EntryArray* res = nullptr;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p.get(); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Array!"; }},
        get_value());

    return *res;
}

void Entry::resize(std::size_t num) { as_array().resize(num); }

Entry Entry::pop_back() { return Entry(as_array().pop_back()); }

Entry Entry::push_back() { return Entry(as_array().push_back()); }

Entry Entry::at(const Path::Segment& p) { return Entry{m_value_, m_path_.join(p)}; }

const Entry Entry::at(const Path::Segment& p) const { return Entry{m_value_, m_path_.join(p)}; }

Entry Entry::at(const Path& path)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>& ref) { as_object().at(path).swap(res.m_value_); },
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) {  entry_value_type(object_p).swap(res.m_value_);res.m_path_=path; },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) {  entry_value_type(array_p).swap(res.m_value_);res.m_path_=path; },
            [&](std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        get_value());

    return std::move(res);
}

const Entry Entry::at(const Path& path) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { entry_value_type(object_p).swap(res.m_value_);res.m_path_=path; },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { entry_value_type(array_p).swap(res.m_value_);res.m_path_=path; },
            [&](const std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

} // namespace sp::db
namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Array, sp::db::Entry::value_type>& array_p) {
                os << "[";

                array_p->for_each([&](int idx, sp::db::entry_value_type const& value) {
                    os << std::endl
                       << std::setw(indent * tab) << " ";
                    fancy_print(os, value, indent + 1, tab);
                    os << ",";
                });

                os << std::endl
                   << std::setw(indent * tab)
                   << "]";
            },
            [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Object, sp::db::Entry::value_type>& object_p) {
                os << "{";

                object_p->for_each([&](const std::string& key, sp::db::entry_value_type const& value) {
                    os << std::endl
                       << std::setw(indent * tab) << " " << key << " : ";
                    fancy_print(os, value, indent + 1, tab);
                    os << ",";
                });

                os << std::endl
                   << std::setw(indent * tab)
                   << "}";
            },
            //    [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Reference, sp::db::Entry::value_type>& ref) {
            //        os << "<" << ref.second.str() << ">";
            //    },                                                                                                                                                                    //
            [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Block, sp::db::Entry::value_type>& blk_p) { fancy_print(os, "<DATA BLOCK>", indent + 1, tab); }, //
            [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Null, sp::db::Entry::value_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },           //
            [&](auto const& ele) { fancy_print(os, ele, indent + 1, tab); }                                                                                                       //
        },
        entry.get_value());

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