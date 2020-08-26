#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{

//----------------------------------------------------------------------------------------------------

// entry_value_type EntryContainer::insert(const Path& path, entry_value_type&& v)
// {
//     VERBOSE << path.str();

//     // EntryContainer* current = this;
//     // auto it = path.begin();
//     // auto ie = path.end();
//     // Path::Segment seg;
//     // do
//     // {
//     //     seg = *it;
//     //     ++it;
//     //     if (it != ie)
//     //     {
//     //         current = current->insert_container(seg);
//     //     }
//     // } while ((it != ie));

//     // return current->at(Path(seg));
//     return entry_value_type{};
// }

// entry_value_type EntryContainer::find(const Path& path) const
// {
//     entry_value_type res;
//     NOT_IMPLEMENTED;

//     // EntryContainer* current = this;
//     // auto it = path.begin();
//     // auto ie = path.end();
//     // Path::Segment seg;
//     // do
//     // {
//     //     seg = *it;
//     //     ++it;
//     //     if (it != ie)
//     //     {
//     //         current = current->insert_container(seg);
//     //     }
//     // } while ((it != ie));

//     // return current->at(Path(seg));
//     return std::move(res);
// }

// void EntryContainer::set_value(const Path& path, entry_value_type&& v)
// {
//     EntryContainer* current = this;
//     auto it = path.begin();
//     auto ie = path.end();
//     Path::Segment seg;
//     do
//     {
//         seg = *it;
//         ++it;
//         if (it != ie)
//         {
//             // current = current->insert_container(seg);
//         }
//     } while ((it != ie));

//     return current->set_value(seg, std::move(v));
// }

// entry_value_type EntryContainer::get_value(const Path& path) const { return find(path); }

// void EntryContainer::remove(const Path& path)
// {
//     NOT_IMPLEMENTED;
//     // find(path.prefix()).remove(path.last());
// }

// std::shared_ptr<EntryObject> EntryContainer::as_object(const Path& path)
// {
//     if (!path.empty())
//     {
//         std::shared_ptr<EntryObject> res;

//         NOT_IMPLEMENTED;
//     }
//     else
//     {
//         return std::dynamic_pointer_cast<EntryObject>(shared_from_this());
//     }
// }

// std::shared_ptr<const EntryObject> EntryContainer::as_object(const Path& path) const
// {
//     if (!path.empty())
//     {
//         std::shared_ptr<const EntryObject> res;

//         NOT_IMPLEMENTED;
//     }
//     else
//     {
//         return std::dynamic_pointer_cast<const EntryObject>(shared_from_this());
//     }
// }

// std::shared_ptr<EntryArray> EntryContainer::as_array(const Path& path)
// {
//     if (!path.empty())
//     {
//         std::shared_ptr<EntryArray> res;

//         NOT_IMPLEMENTED;
//     }
//     else
//     {
//         return std::dynamic_pointer_cast<EntryArray>(shared_from_this());
//     }
// }

// std::shared_ptr<const EntryArray> EntryContainer::as_array(const Path& path) const
// {
//     if (!path.empty())
//     {
//         std::shared_ptr<const EntryArray> res;

//         NOT_IMPLEMENTED;
//     }
//     else
//     {
//         return std::dynamic_pointer_cast<const EntryArray>(shared_from_this());
//     }
// }

//----------------------------------------------------------------------------------------------------
/**
 *  Create 
 */
Entry EntryObject::insert(const Path&, entry_value_type&&)
{
    NOT_IMPLEMENTED;
    return Entry{};
}
/**
 * Modify
 */
void EntryObject::update(const Path&, entry_value_type&& v) { NOT_IMPLEMENTED; }
/**
 * Retrieve
 */
Entry EntryObject::find(const Path& key) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

/**
 *  Delete 
 */
void EntryObject::remove(const Path& path) { NOT_IMPLEMENTED; }

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

    Entry insert(const std::string&, entry_value_type&&) override;

    void set_value(const std::string& key, entry_value_type&& v) override;

    entry_value_type get_value(const std::string& key) const override;

    void remove(const std::string& path) override;

    //------------------------------------------------------------------

    Entry insert(const Path&, entry_value_type&&) override;

    void update(const Path&, entry_value_type&& v = {}) override;

    Entry find(const Path& key) const override;

    void remove(const Path& path) override;
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

Entry EntryObjectDefault::insert(const std::string&, entry_value_type&&)
{
    NOT_IMPLEMENTED;
    return Entry{};
}

entry_value_type EntryObjectDefault::get_value(const std::string& key) const { return entry_value_type(m_container_.at(key)); }

void EntryObjectDefault::set_value(const std::string& key, entry_value_type&& v) { m_container_[key].swap(v); }

void EntryObjectDefault::remove(const std::string& key) { m_container_.erase(m_container_.find(key)); }

Entry EntryObjectDefault::insert(const Path&, entry_value_type&&)
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryObjectDefault::update(const Path&, entry_value_type&& v) { NOT_IMPLEMENTED; }

Entry EntryObjectDefault::find(const Path& key) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryObjectDefault::remove(const Path& path) { NOT_IMPLEMENTED; }

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

Entry EntryArray::insert(const Path&, entry_value_type&&)
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryArray::update(const Path& key, entry_value_type&& v) { NOT_IMPLEMENTED; }

Entry EntryArray::find(const Path& path) const
{
    NOT_IMPLEMENTED;
    return Entry{};
}

void EntryArray::remove(const Path& path) { NOT_IMPLEMENTED; }

//===========================================================================================================
// Entry

//-----------------------------------------------------------------------------------------------------------

namespace _detail
{
entry_value_type insert(entry_value_type& self, const Path& path, entry_value_type v = {})
{
    entry_value_type res;

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Null, entry_value_type>& ref) { NOT_IMPLEMENTED; },
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->insert(path, std::move(v)).value().swap(res); },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->insert(path, std::move(v)).value().swap(res); },
            [&](std::variant_alternative_t<entry_value_type_tags::Block, entry_value_type>& blk_p) { NOT_IMPLEMENTED; },
            [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
        self);

    // std::visit(
    //     sp::traits::overloaded{
    //         [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) { object_p->insert(path, std::move(v)).swap(res); },
    //         [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) { array_p->insert(path, std::move(v)).swap(res); },
    //         [&](std::variant_alternative_t<entry_value_type_tags::Null, entry_value_type>&) {
    //             NOT_IMPLEMENTED;
    //             // m_value_.emplace<value_type_tags::Object>(std::make_shared<EntryObjectDefault>());
    //             // res = std::get<entry_value_type_tags::Object>(m_value_)->insert(m_path_, std::make_shared<EntryObjectDefault>()).as_object();
    //         },
    //         [&](auto&& v) { RUNTIME_ERROR << "Can not convert to Object!"; }},
    //     self);

    return std::move(res);
}

void update(entry_value_type& self, const Path& path, entry_value_type&& v)
{
    // if (!m_path_.empty())
    // {
    //     std::visit(
    //         sp::traits::overloaded{
    //             [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->update(m_path_, std::move(v)); },
    //             [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->update(m_path_, std::move(v)); },
    //             [&](std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
    //             [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
    //         m_value_);
    // }
    // else
    // {
    //     v.swap(m_value_);
    // }
    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->update(path, std::move(v));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->update(path.join(path), std::move(v));
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Null, entry_value_type>&) {
                NOT_IMPLEMENTED;
                // m_value_.emplace<value_type_tags::Object>(std::make_shared<EntryObjectDefault>());
                // res = std::get<entry_value_type_tags::Object>(m_value_)->insert(m_path_, std::make_shared<EntryObjectDefault>()).as_object();
            },
            [&](auto&& v) {
                if (path.empty())
                {
                    // v.swap(self);
                }
                else
                {
                    RUNTIME_ERROR << "illegal path!";
                }
            }},
        self);
}

entry_value_type find(const entry_value_type& self, const Path& path)
{
    // Entry res;
    // std::visit(
    //     sp::traits::overloaded{
    //         [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { Entry{object_p, m_path_.join(path)}.swap(res); },
    //         [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { Entry{array_p, m_path_.join(path)}.swap(res); },
    //         [&](const std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
    //         [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
    //     dynamic_cast<const value_type&>(m_value_));
    // return std::move(res);

    // Entry res;

    entry_value_type res;

    // if (!m_path_.empty())
    // {
    //     std::visit(
    //         sp::traits::overloaded{
    //             [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->find(m_path_).get_value().swap(res); },
    //             [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->find(m_path_).get_value().swap(res); },
    //             [&](const std::variant_alternative_t<value_type_tags::Block, value_type>& blk_p) { NOT_IMPLEMENTED; },
    //             [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
    //         m_value_);

    //     std::move(res);
    // }
    // else
    // {
    //     entry_value_type(m_value_).swap(res);
    // }
    // return std::move(res);

    std::visit(
        sp::traits::overloaded{
            [&](const std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                entry_value_type(object_p->find(path).value()).swap(res);
            },
            [&](const std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                entry_value_type(array_p->find(path).value()).swap(res);
            },
            [&](auto&& v) {
                if (path.empty())
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

void remove(entry_value_type& self, const Path& path)
{

    std::visit(
        sp::traits::overloaded{
            [&](std::variant_alternative_t<entry_value_type_tags::Object, entry_value_type>& object_p) {
                object_p->remove(path);
            },
            [&](std::variant_alternative_t<entry_value_type_tags::Array, entry_value_type>& array_p) {
                array_p->remove(path);
            },
            [&](auto&& v) {
                self.emplace<entry_value_type_tags::Null>(nullptr);
            }},
        self);
}

} // namespace _detail

//-----------------------------------------------------------------------------------------------------------
Entry::Entry(entry_value_type v, Path const& p) : m_value_(std::move(v)), m_path_(p) {}

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
        get_value());

    return res;
}

//-----------------------------------------------------------------------------------------------------------

Entry Entry::at(const Path& path) { return Entry{m_value_, m_path_.join(path)}; }

const Entry Entry::at(const Path& path) const { return Entry{m_value_, m_path_.join(path)}; }

void Entry::set_value(entry_value_type v) { _detail::update(m_value_, m_path_, std::move(v)); }

entry_value_type Entry::get_value() const { return _detail::find(m_value_, m_path_); }

/**
 * TODO: check type of the return of insert
*/
std::shared_ptr<EntryObject> Entry::as_object() { return std::get<entry_value_type_tags::Object>(_detail::insert(m_value_, m_path_, std::make_shared<EntryObjectDefault>())); }

std::shared_ptr<const EntryObject> Entry::as_object() const { return std::const_pointer_cast<const EntryObject>(std::get<entry_value_type_tags::Object>(_detail::find(m_value_, m_path_))); }

std::shared_ptr<EntryArray> Entry::as_array() { return std::get<entry_value_type_tags::Array>(_detail::insert(m_value_, m_path_, std::make_shared<EntryArray>())); }

std::shared_ptr<const EntryArray> Entry::as_array() const { return std::const_pointer_cast<const EntryArray>(std::get<entry_value_type_tags::Array>(_detail::find(m_value_, m_path_))); }

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
        _detail::insert(m_value_, m_path_, std::make_shared<EntryObjectDefault>()));
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
        _detail::find(m_value_, m_path_));
    return std::move(res);
}

void Entry::for_each(std::function<void(const Path::Segment&, entry_value_type&)> const&) { NOT_IMPLEMENTED; }

void Entry::for_each(std::function<void(const Path::Segment&, const entry_value_type&)> const&) const { NOT_IMPLEMENTED; }

// Entry Entry::at(const Path::Segment& p)
// {
//     std::visit(
//         sp::traits::overloaded{
//             [&](std::string const& key) { as_object()->insert(Path(p)).swap(res); },
//             [&](int idx) { entry_value_type(as_array()->at(idx)).swap(res.m_value_); },
//             [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
//         p);
// }

// const Entry Entry::at(const Path::Segment& p) const
// {
//     std::visit(
//         sp::traits::overloaded{
//             [&](std::string const& key) { as_object()->get_value(p).swap(res.m_value_); },
//             [&](int idx) { as_array()->get_value(idx).swap(res.m_value_); },
//             [&](auto&& v) { RUNTIME_ERROR << "illegal index type"; }},
//         p);
// }
// std::shared_ptr<EntryContainer> EntryObjectDefault::insert_container(const std::string& key)
// {
//     std::shared_ptr<EntryContainer> res;

//     std::visit(
//         sp::traits::overloaded{
//             [&](const std::string& k) {
//                 auto p = m_container_.try_emplace(k);
//                 if (!p.second)
//                 {
//                     p.first->second.emplace<value_type_tags::Object>(new EntryObjectDefault);
//                 }
//                 res = std::get<value_type_tags::Object>(p.first->second);
//             },

//             [&](auto&&) { RUNTIME_ERROR << "illegal type! "; }},
//         key);
//     return res;
// }

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