#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"

namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::entry_value_type& v, int indent = 0, int tab = 4);

std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4) { return fancy_print(os, entry.get_value(), indent, tab); }
} // namespace sp::utility

namespace sp::db
{

std::ostream& operator<<(std::ostream& os, Entry const& entry) { return sp::utility::fancy_print(os, entry, 0); }

std::ostream& operator<<(std::ostream& os, entry_value_type const& entry) { return sp::utility::fancy_print(os, entry, 0); }

namespace _detail
{
entry_value_type insert_container(entry_value_type self, const Path& path)
{
    bool success = true;
    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                    object_p->insert(std::get<Path::segment_tags::Key>(*it), entry_value_type{EntryObject::create()}).swap(self);
                },
                [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                    array_p->insert(std::get<Path::segment_tags::Index>(*it), entry_value_type{EntryObject::create()}).swap(self);
                },
                [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>&) {
                    NOT_IMPLEMENTED;
                },
                [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
            self);
    }
    return self;
}

entry_value_type find(entry_value_type self, const Path& path)
{

    bool found = true;

    for (auto it = path.begin(), ie = path.end(); it != ie; ++it)
    {
        switch (self.index())
        {
        case entry_value_type_tags::Object:
            entry_value_type(std::get<entry_value_type_tags::Object>(self)->find(std::get<Path::segment_tags::Key>(*it))).swap(self);
            break;
        case entry_value_type_tags::Array:
            entry_value_type(std::get<entry_value_type_tags::Array>(self)->find(std::get<Path::segment_tags::Index>(*it))).swap(self);
            break;
        default:
            found = false;
            break;
        }

        // std::visit(
        //     sp::traits::overloaded{
        //         [&](const std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
        //             auto tmp = object_p->find(std::get<Path::segment_tags::Key>(*it));
        //             VERBOSE << tmp;
        //             self.swap(tmp);
        //         },
        //         [&](const std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->find(std::get<Path::segment_tags::Index>(*it)).swap(self); },
        //         [&](auto&& v) { NOT_IMPLEMENTED; }},
        //     self);
    }
    return found ? self : entry_value_type{};
}

entry_value_type insert(entry_value_type self, entry_value_type&& v, const Path& path)
{

    insert_container(self, path.prefix()).swap(self);

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->insert(std::get<Path::segment_tags::Key>(path.last()), std::move(v)).swap(self);
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->insert(std::get<Path::segment_tags::Index>(path.last()), std::move(v)).swap(self);
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
        self);

    return self;
}

void update(entry_value_type self, entry_value_type&& v, const Path& path)
{
    insert_container(self, path.prefix()).swap(self);

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->update(std::get<Path::segment_tags::Key>(path.last()), std::move(v));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->update(std::get<Path::segment_tags::Index>(path.last()), std::move(v));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
        self);
}

void remove(entry_value_type self, const Path& path)
{
    find(self, path.prefix()).swap(self);

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->remove(std::get<Path::segment_tags::Key>(path.last()));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->remove(std::get<Path::segment_tags::Index>(path.last()));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>&) {
                NOT_IMPLEMENTED;
            },
            [&](auto&& v) { RUNTIME_ERROR << "Can not insert value to non-container object!"; }},
        self);
}

} // namespace _detail

//----------------------------------------------------------------------------------------------------

entry_value_type EntryObject::insert(entry_value_type v, const Path& path) { return _detail::insert(entry_value_type{shared_from_this()}, std::move(v), path); }

entry_value_type EntryObject::find(const Path& path) const { return _detail::find(entry_value_type{const_cast<EntryObject*>(this)->shared_from_this()}, path); }

void EntryObject::update(entry_value_type v, const Path& path) { _detail::update(entry_value_type{shared_from_this()}, std::move(v), path); }

void EntryObject::remove(const Path& path) { _detail::remove(entry_value_type{const_cast<EntryObject*>(this)->shared_from_this()}, path); }

void EntryObject::merge(const EntryObject&) { NOT_IMPLEMENTED; }

void EntryObject::patch(const EntryObject&) { NOT_IMPLEMENTED; }

void EntryObject::update(const EntryObject&) { NOT_IMPLEMENTED; }
//----------------------------------------------------------------------------------------------------

class EntryObjectDefault : public EntryObject
{
private:
    std::map<std::string, entry_value_type> m_container_;

public:
    typedef EntryObjectDefault this_type;
    typedef Entry::value_type_tags value_type_tags;

    EntryObjectDefault() = default;

    EntryObjectDefault(const this_type& other) : m_container_(other.m_container_) {}

    EntryObjectDefault(this_type&& other) : m_container_(std::move(other.m_container_)) {}

    virtual ~EntryObjectDefault() = default;

    std::unique_ptr<EntryObject> copy() const override { return std::unique_ptr<EntryObject>(new EntryObjectDefault(*this)); }

    size_t size() const override { return m_container_.size(); }

    void clear() override { return m_container_.clear(); }

    Entry at(const Path& path) override;

    Entry at(const Path& path) const override;

    Cursor<entry_value_type> children() override;

    Cursor<const entry_value_type> children() const override;

    void for_each(std::function<void(const std::string&, entry_value_type&)> const&) override;

    void for_each(std::function<void(const std::string&, const entry_value_type&)> const&) const override;
    //------------------------------------------------------------------

    entry_value_type insert(const std::string&, entry_value_type) override;

    entry_value_type find(const std::string&) const override;

    void update(const std::string&, entry_value_type) override;

    void remove(const std::string&) override;

    //------------------------------------------------------------------
};

Entry EntryObjectDefault::at(const Path& path) { return Entry{entry_value_type{std::in_place_index_t<entry_value_type_tags::Object>(), std::dynamic_pointer_cast<EntryObject>(shared_from_this())}, path}; };

Entry EntryObjectDefault::at(const Path& path) const { return Entry{entry_value_type{std::in_place_index_t<entry_value_type_tags::Object>(), std::dynamic_pointer_cast<EntryObject>(const_cast<EntryObjectDefault*>(this)->shared_from_this())}, path}; }

Cursor<entry_value_type> EntryObjectDefault::children()
{
    return Cursor<entry_value_type>{};
}

Cursor<const entry_value_type> EntryObjectDefault::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
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

entry_value_type EntryObjectDefault::insert(const std::string& key, entry_value_type v) { return m_container_.emplace(key, std::move(v)).first->second; }

entry_value_type EntryObjectDefault::find(const std::string& key) const { return m_container_.at(key); }

void EntryObjectDefault::update(const std::string& key, entry_value_type v) { m_container_[key].swap(v); }

void EntryObjectDefault::remove(const std::string& key) { m_container_.erase(m_container_.find(key)); }

//------------------------------------------------------------------

std::shared_ptr<EntryObject> EntryObject::create(const std::string& backend) { return std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryObjectDefault>()); }

//==========================================================================================
// EntryArray

std::shared_ptr<EntryArray> EntryArray::create(const std::string& backend) { return std::make_shared<EntryArray>(); }

void EntryArray::resize(std::size_t num) { m_container_.resize(num); }

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

// Entry EntryArray::at(const Path& path)
// {
//     return Entry{std::dynamic_pointer_cast<EntryArray>(shared_from_this()), path};
// }

// Entry EntryArray::at(const Path& path) const
// {
//     return const_cast<EntryArray*>(this)->at(path);
// }

void EntryArray::for_each(std::function<void(int, entry_value_type&)> const& visitor)
{
    for (int i = 0, s = m_container_.size(); i < s; ++i)
    {
        visitor(i, m_container_[i]);
    }
}

void EntryArray::for_each(std::function<void(int, const entry_value_type&)> const& visitor) const
{
    for (int i = 0, s = m_container_.size(); i < s; ++i)
    {
        visitor(i, m_container_[i]);
    }
}

entry_value_type EntryArray::find(int idx) const { return m_container_.at(idx); }

entry_value_type EntryArray::insert(int idx, entry_value_type&& v)
{
    if (m_container_[idx].index() == entry_value_type_tags::Null)
    {
        v.swap(m_container_[idx]);
    }
    return m_container_[idx];
};

void EntryArray::update(int idx, entry_value_type&& v) { m_container_[idx].swap(v); }

void EntryArray::remove(int idx) { NOT_IMPLEMENTED; };

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

Entry EntryArray::push_back(entry_value_type v)
{
    m_container_.emplace_back(std::move(v));

    return Entry{std::dynamic_pointer_cast<EntryArray>(shared_from_this()), Path(m_container_.size() - 1)};
}

Entry EntryArray::pop_back()
{
    entry_value_type res(m_container_.back());
    m_container_.pop_back();
    return Entry{res, Path{}};
}

entry_value_type EntryArray::insert(entry_value_type v, const Path& path)
{

    if (v.index() == entry_value_type_tags::Object && (std::get<entry_value_type_tags::Object>(v) == nullptr))
    {
        v.emplace<entry_value_type_tags::Object>(EntryObject::create());
    }
    else if (v.index() == entry_value_type_tags::Array && (std::get<entry_value_type_tags::Array>(v) == nullptr))
    {
        v.emplace<entry_value_type_tags::Array>(EntryArray::create());
    }

    return _detail::insert(entry_value_type{shared_from_this()}, std::move(v), path);
}

entry_value_type EntryArray::find(const Path& path) const { return _detail::find(entry_value_type{const_cast<EntryArray*>(this)->shared_from_this()}, path); }

void EntryArray::update(entry_value_type v, const Path& path) { _detail::update(entry_value_type{shared_from_this()}, std::move(v), path); }

void EntryArray::remove(const Path& path) { _detail::remove(entry_value_type{shared_from_this()}, path); }

//===========================================================================================================
// Entry

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
Entry::Entry(entry_value_type v, const Path& p) : m_value_(std::move(v)), m_path_(p) {}

Entry::Entry(const Entry& other) : m_value_(other.m_value_), m_path_(other.m_path_) {}

Entry::Entry(Entry&& other) : m_value_(std::move(other.m_value_)), m_path_(std::move(other.m_path_)) {}

void Entry::swap(Entry& other)
{
    std::swap(m_value_, other.m_value_);
    std::swap(m_path_, other.m_path_);
}

std::size_t Entry::type() const { return fetch().index(); }

void Entry::reset()
{
    m_value_.emplace<std::nullptr_t>(nullptr);
    m_path_.clear();
}

bool Entry::is_null() const { return type() == value_type_tags::Null; }

bool Entry::empty() const { return is_null() || size() == 0; }

size_t Entry::size() const
{
    size_t res = 0;
    auto tmp = fetch();

    switch (tmp.index())
    {
    case entry_value_type_tags::Object:
        res = std::get<entry_value_type_tags::Object>(tmp)->size();
        break;
    case entry_value_type_tags::Array:
        res = std::get<entry_value_type_tags::Array>(tmp)->size();
        break;
    default:
        res = 0;
    }
    return res;
}

//-----------------------------------------------------------------------------------------------------------

void Entry::set_value(entry_value_type v) { assign(std::move(v)); }

entry_value_type Entry::get_value() const { return fetch(); }

EntryObject& Entry::as_object() { return *std::get<entry_value_type_tags::Object>(fetch(entry_value_type{EntryObject::create()})); }

const EntryObject& Entry::as_object() const { return *std::const_pointer_cast<const EntryObject>(std::get<entry_value_type_tags::Object>(fetch())); }

EntryArray& Entry::as_array() { return *std::get<entry_value_type_tags::Array>(fetch(entry_value_type{EntryArray::create()})); }

const EntryArray& Entry::as_array() const { return *std::const_pointer_cast<const EntryArray>(std::get<entry_value_type_tags::Array>(fetch())); }

entry_value_type& Entry::root()
{

    switch (m_value_.index())
    {
    case entry_value_type_tags::Null:
        m_value_.emplace<entry_value_type_tags::Object>(EntryObject::create());

        break;
    case entry_value_type_tags::Object:
    case entry_value_type_tags::Array:
        break;
    case entry_value_type_tags::Block:
        NOT_IMPLEMENTED;
        break;
    default:
        RUNTIME_ERROR << "Entry is not a root!";
        break;
    }

    return m_value_;
}

const entry_value_type& Entry::root() const
{
    if (m_value_.index() != entry_value_type_tags::Object && m_value_.index() != entry_value_type_tags::Array)
    {
        RUNTIME_ERROR << "Entry is not a root!";
    }
    return m_value_;
}
//-----------------------------------------------------------------------------------------------------------

void Entry::resize(std::size_t num) { as_array().resize(num); }

Entry Entry::pop_back() { return as_array().pop_back(); }

Entry Entry::push_back(entry_value_type v) { return as_array().push_back(std::move(v)); }

Cursor<entry_value_type> Entry::Entry::children()
{
    Cursor<entry_value_type> res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->children().swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
            [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
        fetch());
    return std::move(res);
}

Cursor<const entry_value_type> Entry::Entry::children() const
{
    Cursor<const entry_value_type> res;
    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { std::const_pointer_cast<const EntryObject>(object_p)->children().swap(res); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { std::const_pointer_cast<const EntryArray>(array_p)->children().swap(res); },
            [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
        fetch());
    return std::move(res);
}

void Entry::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) { NOT_IMPLEMENTED; }

void Entry::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const { NOT_IMPLEMENTED; }

entry_value_type Entry::fetch(entry_value_type default_value)
{

    entry_value_type res;

    if (m_path_.empty())
    {
        if (m_value_.index() == entry_value_type_tags::Null)
        {
            m_value_.swap(default_value);
        }

        entry_value_type(m_value_).swap(res);
    }
    else
    {

        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->insert(std::move(default_value), m_path_).swap(res); },
                [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->insert(std::move(default_value), m_path_).swap(res); },
                [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&&) { RUNTIME_ERROR << "Try insert value to non-container entry!"; }},
            root());
    }

    return std::move(res);
}

entry_value_type Entry::fetch() const
{

    entry_value_type res;

    if (m_path_.empty())
    {
        entry_value_type(m_value_).swap(res);
    }
    else
    {

        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->find(m_path_).swap(res); },
                [&](const std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->find(m_path_).swap(res); },
                [&](const std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&&) { RUNTIME_ERROR << "Try access non-container entry!"; }},
            root());
    }
    return std::move(res);
}

void Entry::assign(entry_value_type&& v)
{
    if (m_path_.empty())
    {
        m_value_.swap(v);
    }
    else
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->update(std::move(v), m_path_); },
                [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->update(std::move(v), m_path_); },
                [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk) { NOT_IMPLEMENTED; },
                [&](auto&&) { RUNTIME_ERROR << "Try to insert value to non-container entry!"; }},
            root());
    }
}

// void Entry::remove(const Path& p)
// {
//     Path path = m_path_.join(p);
//     if (path.empty())
//     {
//         m_value_.emplace<entry_value_type_tags::Null>();
//     }
//     else
//     {
//         std::visit(
//             sp::traits::overloaded{
//                 [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->remove(path); },
//                 [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->remove(path); },
//                 [&](auto&&) {}},
//             m_value_);
//     }
// }

} // namespace sp::db

namespace sp::utility
{

std::ostream& fancy_print(std::ostream& os, const sp::db::entry_value_type& v, int indent, int tab)
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
                   << std::setw(indent * tab) << " "
                   << "]";
            },
            [&](const std::variant_alternative_t<sp::db::Entry::value_type_tags::Object, sp::db::Entry::value_type>& object_p) {
                os << "{";

                object_p->for_each(
                    [&](const sp::db::Path::Segment& key, sp::db::entry_value_type const& value) {
                        os << std::endl
                           << std::setw(indent * tab) << " "
                           << "\"" << std::get<std::string>(key) << "\" : ";
                        fancy_print(os, value, indent + 1, tab);
                        os << ",";
                    });

                os << std::endl
                   << std::setw(indent * tab) << " "
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

} // namespace sp::utility
