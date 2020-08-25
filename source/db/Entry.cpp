#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{

//----------------------------------------------------------------------------------------------------

Entry EntryContainer::insert(const Path& path)
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
            current = current->insert_container(seg);
        }
    } while ((it != ie));

    return current->at(Path(seg));
}

const entry_value_type EntryContainer::find(const Path& path) const
{
    entry_value_type res;
    NOT_IMPLEMENTED;

    // EntryContainer* current = this;
    // auto it = path.begin();
    // auto ie = path.end();
    // Path::Segment seg;
    // do
    // {
    //     seg = *it;
    //     ++it;
    //     if (it != ie)
    //     {
    //         current = current->insert_container(seg);
    //     }
    // } while ((it != ie));

    // return current->at(Path(seg));
    return std::move(res);
}

void EntryContainer::set_value(const Path& path, entry_value_type&& v)
{
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
            current = current->insert_container(seg);
        }
    } while ((it != ie));

    return current->set_value(seg, std::move(v));
}

entry_value_type EntryContainer::get_value(const Path& path) const { return find(path); }

void EntryContainer::remove(const Path& path)
{
    NOT_IMPLEMENTED;
    // find(path.prefix()).remove(path.last());
}

//----------------------------------------------------------------------------------------------------
size_t EntryObject::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

void EntryObject::clear() { NOT_IMPLEMENTED; }

Entry EntryObject::at(const Path& path) { return Entry{std::dynamic_pointer_cast<EntryObject>(shared_from_this()), path}; }

Entry EntryObject::at(const Path& path) const { return const_cast<EntryObject*>(this)->at(path); }

Cursor<entry_value_type> EntryObject::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>{};
}

Cursor<const entry_value_type> EntryObject::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

void EntryObject::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const& visitor) { NOT_IMPLEMENTED; }

void EntryObject::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const& visitor) const { NOT_IMPLEMENTED; }

//----------------------------------------------------------------------------------------------------

EntryContainer* EntryObject::insert_container(const Path::Segment& key) { return nullptr; }

void EntryObject::set_value(const Path::Segment& key, entry_value_type&&)
{
    NOT_IMPLEMENTED;
}

entry_value_type EntryObject::get_value(const Path::Segment& key) const
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

    Cursor<entry_value_type> children() override;

    Cursor<const entry_value_type> children() const override;

    void for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) override;

    void for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const override;

    EntryContainer* insert_container(const Path::Segment& key) override;

    void set_value(const Path::Segment& key, entry_value_type&& v = {}) override;

    entry_value_type get_value(const Path::Segment& key) const override;

    void remove(const Path::Segment& path) override;

    void merge(const EntryObject&) override;

    void patch(const EntryObject&) override;

    void update(const EntryObject& patch) override;
};

Cursor<entry_value_type> EntryObjectDefault::children()
{
    return Cursor<entry_value_type>{};
}

Cursor<const entry_value_type> EntryObjectDefault::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

void EntryObjectDefault::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const& visitor)
{

    for (auto&& item : m_container_)
    {
        visitor(Path::Segment(item.first), item.second);
    }
}

void EntryObjectDefault::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const& visitor) const
{
    for (auto&& item : m_container_)
    {
        visitor(Path::Segment(item.first), item.second);
    }
}

EntryContainer* EntryObjectDefault::insert_container(const Path::Segment& key)
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

// {
//     const EntryContainer* res;

//     std::visit(
//         sp::traits::overloaded{
//             [&](const std::string& k) {
//                 auto p = m_container_.find(k);
//                 if (p != m_container_.end() && p->second.index() == value_type_tags::Object)
//                 {
//                     res = std::get<value_type_tags::Object>(p->second).get();
//                 }
//                 res = std::get<value_type_tags::Object>(p->second).get();
//             },
//             [&](auto&&) { RUNTIME_ERROR << "illegal type! "; }},
//         key);
//     return res;
// }

entry_value_type EntryObjectDefault::get_value(const Path::Segment& key) const
{
    entry_value_type res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::string& k) { entry_value_type(m_container_.at(k)).swap(res); },
            [&](auto&&) { RUNTIME_ERROR << "illegal key type! "; }},
        key);
    return std::move(res);
}

void EntryObjectDefault::set_value(const Path::Segment& key, entry_value_type&& v)
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

void EntryArray::resize(std::size_t num)
{
    VERBOSE << num;
    m_container_.resize(num);
}

size_t EntryArray::size() const { return m_container_.size(); }

void EntryArray::clear() { m_container_.clear(); }

Cursor<entry_value_type> EntryArray::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>(); /*(m_container_.begin(), m_container_.end());*/
}

Cursor<const entry_value_type> EntryArray::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>(); /*(m_container_.cbegin(), m_container_.cend());*/
}

Entry EntryArray::at(const Path& path)
{
    return Entry{std::dynamic_pointer_cast<EntryArray>(shared_from_this()), path};
}

Entry EntryArray::at(const Path& path) const
{
    return const_cast<EntryArray*>(this)->at(path);
}

void EntryArray::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const& visitor)
{

    NOT_IMPLEMENTED;
}

void EntryArray::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const& visitor) const
{
    NOT_IMPLEMENTED;
}

EntryContainer* EntryArray::insert_container(const Path::Segment& key)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

void EntryArray::set_value(const Path::Segment& path, entry_value_type&& v)
{
    m_container_[std::get<int>(path)].swap(v);
}

entry_value_type EntryArray::get_value(const Path::Segment& path) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

void EntryArray::remove(const Path::Segment& path)
{
    NOT_IMPLEMENTED;
}

entry_value_type EntryArray::get_value(int idx) { return (m_container_.at(idx)); }

entry_value_type EntryArray::get_value(int idx) const { return (m_container_.at(idx)); }

entry_value_type EntryArray::slice(int start, int stop, int step)
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

entry_value_type EntryArray::slice(int start, int stop, int step) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

Entry EntryArray::push_back()
{
    m_container_.emplace_back();

    return Entry{std::dynamic_pointer_cast<EntryArray>(shared_from_this()), Path(m_container_.size() - 1)};
}

Entry EntryArray::pop_back()
{
    entry_value_type res(m_container_.back());
    m_container_.pop_back();
    return Entry{res, Path{}};
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
        m_value_);
    return res;
}

Cursor<entry_value_type> Entry::Entry::children()
{
    NOT_IMPLEMENTED;
    Cursor<entry_value_type> res;

    // std::visit(
    //     sp::traits::overloaded{
    //         [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->children().swap(res); },
    //         [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
    //         [&](auto&&) {}},
    //     m_value_);
    return std::move(res);
}

Cursor<const entry_value_type> Entry::Entry::children() const
{
    Cursor<const entry_value_type> res;
    NOT_IMPLEMENTED;
    // std::visit(
    //     sp::traits::overloaded{
    //         [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->children().swap(res); },
    //         [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
    //         [&](auto&&) { NOT_IMPLEMENTED; }},
    //     m_value_);
    return std::move(res);
}

void Entry::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) { NOT_IMPLEMENTED; }

void Entry::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const { NOT_IMPLEMENTED; }

Entry Entry::at(const Path& path)
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>& ref) { as_object().insert(m_path_.join(path)).swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { Entry{object_p, m_path_.join(path)}.swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { Entry{array_p, m_path_.join(path)}.swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        m_value_);

    return std::move(res);
}

const Entry Entry::at(const Path& path) const
{
    Entry res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { Entry{object_p, m_path_.join(path)}.swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { Entry{array_p, m_path_.join(path)}.swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        dynamic_cast<const value_type&>(m_value_));
    return std::move(res);
}

//-----------------------------------------------------------------------------------------------------------

void Entry::set_value(entry_value_type&& v)
{
    if (!m_path_.empty())
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->set_value(m_path_, std::move(v)); },
                [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->set_value(m_path_, std::move(v)); },
                [&](std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
                [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
            m_value_);
    }
    else
    {
        v.swap(m_value_);
    }
}

entry_value_type Entry::get_value() const
{
    if (!m_path_.empty())
    {
        entry_value_type res;
        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->get_value(m_path_).swap(res); },
                [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->get_value(m_path_).swap(res); },
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

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& array_p) {},
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>&) { m_value_.emplace<value_type_tags::Object>(std::make_shared<EntryObjectDefault>()); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        m_value_);

    return *std::get<entry_value_type_tags::Object>(m_value_);
}

const EntryObject& Entry::as_object() const
{
    if (m_value_.index() != value_type_tags::Object)
    {
        RUNTIME_ERROR << "Can not convert to Object!";
    }
    return *std::get<entry_value_type_tags::Object>(m_value_);
}

EntryArray& Entry::as_array()
{
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) {},
            [&](std::variant_alternative_t<value_type_tags::Null, value_type>&) { m_value_.emplace<value_type_tags::Array>(std::make_shared<EntryArray>()); },
            [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
        m_value_);

    return *std::get<entry_value_type_tags::Array>(m_value_);
}

const EntryArray& Entry::as_array() const
{
    if (m_value_.index() != value_type_tags::Array)
    {
        RUNTIME_ERROR << "Can not convert to Object!";
    }

    return *std::get<entry_value_type_tags::Array>(m_value_);
}

//-----------------------------------------------------------------------------------------------------------
void Entry::resize(std::size_t num)
{
    as_array().resize(num);
}

Entry Entry::pop_back() { return as_array().pop_back(); }

Entry Entry::push_back() { return as_array().push_back(); }

// Entry Entry::at(const Path::Segment& p)
// {
//     std::visit(
//         sp::traits::overloaded{
//             [&](std::string const& key) { as_object().insert(Path(p)).swap(res); },
//             [&](int idx) { entry_value_type(as_array().at(idx)).swap(res.m_value_); },
//             [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
//         p);
// }

// const Entry Entry::at(const Path::Segment& p) const
// {
//     std::visit(
//         sp::traits::overloaded{
//             [&](std::string const& key) { as_object().get_value(p).swap(res.m_value_); },
//             [&](int idx) { as_array().get_value(idx).swap(res.m_value_); },
//             [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
//         p);
// }
} // namespace sp::db

namespace sp::utility
{

std::ostream& fancy_print(std::ostream& os, const sp::db::entry_value_type& v, int indent = 0, int tab = 4)
{
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Array, sp::db::Entry::value_type>& array_p) {
                os << "[";

                array_p->for_each([&](const sp::db::Path::Segment&, sp::db::entry_value_type const& value) {
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

                object_p->for_each(
                    [&](const sp::db::Path::Segment& key, sp::db::entry_value_type const& value) {
                        os << std::endl
                           << std::setw(indent * tab) << " " << std::get<std::string>(key) << " : ";
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
        v);

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
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    return fancy_print(os, entry.get_value(), indent, tab);
}
} // namespace sp::utility

namespace sp::db
{
std::ostream& operator<<(std::ostream& os, Entry const& entry) { return sp::utility::fancy_print(os, entry, 0); }
} // namespace sp::db