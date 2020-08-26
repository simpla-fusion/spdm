#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{

namespace _detail
{

template <typename IT>
entry_value_type insert(entry_value_type& self, entry_value_type&& v, IT it, IT ie)
{

    if (it == ie)
    {
        entry_value_type(v).swap(self);
        return v;
    }
    else if (self.index() == entry_value_type_tags::Null)
    {
        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) { self.emplace<entry_value_type_tags::Object>(EntryObject::create()); },
                [&](const std::variant_alternative_t<Path::segment_tags::Index, Path::Segment>& idx) { self.emplace<entry_value_type_tags::Array>(EntryArray::create()); },
                [&](auto&&) { NOT_IMPLEMENTED; }},
            *it);
    }
    else if (self.index() > entry_value_type_tags::Array)
    {
        RUNTIME_ERROR << "illegal type";
    }

    entry_value_type* current = &self;

    Path::Segment key = *it;

    while (it != ie)
    {
        entry_value_type tmp;

        std::visit(
            sp::traits::overloaded{
                [&](const std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) { tmp.emplace<entry_value_type_tags::Object>(nullptr); },
                [&](const std::variant_alternative_t<Path::segment_tags::Index, Path::Segment>& idx) { tmp.emplace<entry_value_type_tags::Array>(nullptr); },
                [&](auto&&) { NOT_IMPLEMENTED; }},
            *it);

        if (current->index() == entry_value_type_tags::Null)
        {
        }

        // std::visit(
        //     sp::traits::overloaded{
        //         [&](const std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
        //             object_p->insert(std::get<Path::segment_tags::Key>(*it), entry_value_type{std::in_place_index_t<entry_value_type_tags::Object>()}).swap(*current);
        //         },
        //         [&](const std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
        //             array_p->insert(std::get<Path::segment_tags::Index>(*it), entry_value_type{std::in_place_index_t<entry_value_type_tags::Array>()}).swap(*current);
        //         },
        //         [&](auto&&) { NOT_IMPLEMENTED; }},
        //     *current);

        //     // if (current.index() == entry_value_type_tags::Object && key.index() == Path::segment_tags::Key)
        //     // {
        //     //     current = std::get<entry_value_type_tags::Object>(current)->insert(std::get<Path::segment_tags::Key>(seg), entry_value_type{});
        //     // }

        //     std::visit(
        //         sp::traits::overloaded{
        //             [&](const std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) {

        //             },
        //             [&](int idx) {},
        //             [&](auto&&) { NOT_IMPLEMENTED; }

        //         },
        //         *it);
        //     key = *it;
    }

    current->swap(v);

    return entry_value_type(*current);
}

template <typename IT>
void update(entry_value_type& self, entry_value_type&& v, IT it, IT ie)
{
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->update(Path(it, ie), std::move(v));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->update(Path(it, ie), std::move(v));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Null, entry_value_type>&) {
                insert(self, std::move(v), it, ie);
            },
            [&](auto&& v) {
                if (it == ie)
                {
                    entry_value_type(v).swap(self);
                }
                else
                {
                    RUNTIME_ERROR << "illegal path!";
                }
            }},
        self);
}

template <typename IT>
void remove(entry_value_type& self, IT it, IT ie)
{

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->remove(Path(it, ie));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->remove(Path(it, ie));
            },
            [&](auto&& v) {
                self.emplace<entry_value_type_tags::Null>(nullptr);
            }},
        self);
}

template <typename IT>
entry_value_type find(const entry_value_type& self, IT it, IT ie)
{

    entry_value_type res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->find(Path{it, ie}).swap(res);
            },
            [&](const std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->find(Path{it, ie}).swap(res);
            },
            [&](auto&& v) {
                if (it == ie)
                {
                    entry_value_type(self).swap(res);
                }
                // else
                // {
                //     RUNTIME_ERROR << "illegal path!";
                // }
            }},
        self);

    return std::move(res);
}

template <int CONTAINER_IDX, typename IT>
auto insert_container(entry_value_type& root, IT ib, IT ie)
{
    typedef std::variant_alternative_t<CONTAINER_IDX, entry_value_type> container_type;
    container_type res;
    // Entry res;

    // Entry parent{root, path.prefix()};

    // std::visit(
    //     sp::traits::overloaded{
    //         [&](const std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) {
    //             auto object_p = parent.as_object();
    //             object_p->set_value(key, std::move(v));
    //             Entry{entry_value_type(object_p), Path(key)}.swap(res);
    //         },
    //         [&](const std::variant_alternative_t<Path::segment_tags::Index, Path::Segment>& idx) {
    //             auto array_p = parent.as_array();
    //             array_p->set_value(idx, std::move(v));
    //             Entry{entry_value_type(array_p), Path(idx)}.swap(res);
    //         },
    //         [&](const std::variant_alternative_t<Path::segment_tags::Slice, Path::Segment>& key) { NOT_IMPLEMENTED; },
    //         [&](auto&&) { NOT_IMPLEMENTED; }},
    //     path.last());

    return res;
}
} // namespace _detail

//----------------------------------------------------------------------------------------------------

entry_value_type EntryObject::insert(const Path& path, entry_value_type v)
{
    entry_value_type self{shared_from_this()};
    return _detail::insert(self, std::move(v), path.begin(), path.end());
}

entry_value_type EntryObject::find(const Path& path) const
{
    entry_value_type self{const_cast<EntryObject*>(this)->shared_from_this()};
    return _detail::find(self, path.begin(), path.end());
}

void EntryObject::update(const Path& path, entry_value_type v)
{
    entry_value_type self{shared_from_this()};
    _detail::update(self, std::move(v), path.begin(), path.end());
}

void EntryObject::remove(const Path& path)
{
    entry_value_type self{const_cast<EntryObject*>(this)->shared_from_this()};
    _detail::remove(self, path.begin(), path.end());
}

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

    void set_value(const std::string&, entry_value_type) override;

    entry_value_type get_value(const std::string&) const override;

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

entry_value_type EntryObjectDefault::get_value(const std::string& key) const { return entry_value_type(m_container_.at(key)); }

void EntryObjectDefault::set_value(const std::string& key, entry_value_type v) { m_container_[key].swap(v); }

void EntryObjectDefault::remove(const std::string& key) { m_container_.erase(m_container_.find(key)); }

//------------------------------------------------------------------

std::shared_ptr<EntryObject> EntryObject::create(const std::string& backend) { return std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryObjectDefault>()); }

//==========================================================================================
// EntryArray
std::shared_ptr<EntryArray> EntryArray::create(const std::string& backend) { return std::make_shared<EntryArray>(); }

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

    NOT_IMPLEMENTED;
}

void EntryArray::for_each(std::function<void(int, const entry_value_type&)> const& visitor) const
{
    NOT_IMPLEMENTED;
}

void EntryArray::set_value(int idx, entry_value_type&& v) { m_container_[idx].swap(v); }

entry_value_type EntryArray::get_value(int idx) const { return m_container_.at(idx); }

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

entry_value_type EntryArray::insert(const Path& path, entry_value_type v)
{
    entry_value_type self{shared_from_this()};
    return _detail::insert(self, std::move(v), path.begin(), path.end());
}
entry_value_type EntryArray::find(const Path& path) const
{
    entry_value_type self{const_cast<EntryArray*>(this)->shared_from_this()};
    return _detail::find(self, path.begin(), path.end());
}

void EntryArray::update(const Path& path, entry_value_type v)
{
    entry_value_type self{shared_from_this()};
    _detail::update(self, std::move(v), path.begin(), path.end());
}

void EntryArray::remove(const Path& path)
{
    entry_value_type self{shared_from_this()};
    _detail::remove(self, path.begin(), path.end());
}

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

void Entry::reset() { m_value_.emplace<std::nullptr_t>(nullptr); }

bool Entry::is_null() const { return type() == value_type_tags::Null; }

bool Entry::empty() const { return is_null() || size() == 0; }

size_t Entry::size() const
{
    size_t res = 0;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { res = object_p->size(); },
            [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { res = array_p->size(); },
            [&](auto&&) { res = 0; }},
        _detail::find(m_value_, m_path_.begin(), m_path_.end()));

    return res;
}

//-----------------------------------------------------------------------------------------------------------

void Entry::set_value(entry_value_type v) { _detail::update(m_value_, std::move(v), m_path_.begin(), m_path_.end()); }

entry_value_type Entry::get_value() const { return _detail::find(m_value_, m_path_.begin(), m_path_.end()); }

/**
 * TODO: check type of the return of insert
*/
std::shared_ptr<EntryObject> Entry::as_object() { return std::get<entry_value_type_tags::Object>(insert(Path{}, entry_value_type{std::in_place_index_t<entry_value_type_tags::Object>()})); }

std::shared_ptr<const EntryObject> Entry::as_object() const { return std::const_pointer_cast<const EntryObject>(std::get<entry_value_type_tags::Object>(find(Path{}))); }

std::shared_ptr<EntryArray> Entry::as_array() { return std::get<entry_value_type_tags::Array>(insert(Path{}, entry_value_type{std::in_place_index_t<entry_value_type_tags::Array>()})); }

std::shared_ptr<const EntryArray> Entry::as_array() const { return std::const_pointer_cast<const EntryArray>(std::get<entry_value_type_tags::Array>(find(Path{}))); }

//-----------------------------------------------------------------------------------------------------------
void Entry::resize(std::size_t num) { as_array()->resize(num); }

Entry Entry::pop_back() { return as_array()->pop_back(); }

Entry Entry::push_back() { return as_array()->push_back(); }

Cursor<entry_value_type> Entry::Entry::children()
{
    Cursor<entry_value_type> res;
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->children().swap(res); },
            [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
            [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
        _detail::insert(m_value_, std::make_shared<EntryObjectDefault>(), m_path_.begin(), m_path_.end()));
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
        _detail::find(m_value_, m_path_.begin(), m_path_.end()));
    return std::move(res);
}

void Entry::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) { NOT_IMPLEMENTED; }

void Entry::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const { NOT_IMPLEMENTED; }

entry_value_type Entry::insert(const Path& p, entry_value_type v)
{
    Path path = m_path_.join(p);

    entry_value_type res;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->insert(path, std::move(v)).swap(res); },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->insert(path, std::move(v)).swap(res); },
            [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk) { NOT_IMPLEMENTED; },
            [&](std::variant_alternative_t<entry_value_type_tags::Null, entry_value_type>&) { _detail::insert(m_value_, std::move(v), p.begin(), p.end()).swap(res); },
            [&](auto&&) { RUNTIME_ERROR << "Try insert value to non-container entry!"; }},
        m_value_);

    return std::move(res);
}

entry_value_type Entry::find(const Path& p) const
{
    Path path = m_path_.join(p);

    entry_value_type res;

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->find(path).swap(res); },
            [&](const std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->find(path).swap(res); },
            [&](const std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk) { NOT_IMPLEMENTED; },
            [&](auto&&) {
                if (path.empty())
                {
                    entry_value_type(m_value_).swap(res);
                }
                else
                {
                    RUNTIME_ERROR << "Try access non-container entry!";
                }
            }},
        m_value_);

    return std::move(res);
}

void Entry::update(const Path& p, entry_value_type v)
{
    Path path = m_path_.join(p);

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->update(path, std::move(v)); },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->update(path, std::move(v)); },
            [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk) { NOT_IMPLEMENTED; },
            [&](std::variant_alternative_t<entry_value_type_tags::Null, entry_value_type>&) { _detail::update(m_value_, std::move(v), p.begin(), p.end()); },
            [&](auto&&) {
                if (path.empty())
                {
                    v.swap(m_value_);
                }
                else
                {
                    RUNTIME_ERROR << "Try to insert value to non-container entry!";
                }
            }

        },
        m_value_);
}

void Entry::remove(const Path& p)
{
    Path path = m_path_.join(p);
    if (!path.empty())
    {
        std::visit(
            sp::traits::overloaded{
                [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->remove(path); },
                [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->remove(path); },
                [&](auto&&) {}},
            m_value_);
    }
}

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