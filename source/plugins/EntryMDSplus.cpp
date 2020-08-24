#include "../db/Entry.h"
#include "../db/EntryPlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>

namespace sp::db
{
struct mdsplus_node
{
};
typedef EntryObjectPlugin<mdsplus_node> EntryObjectMDSPLUS;

template <>
EntryObjectMDSPLUS::EntryObjectPlugin(const mdsplus_node& container) : m_container_(container) {}

template <>
EntryObjectMDSPLUS::EntryObjectPlugin(mdsplus_node&& container) : m_container_(std::move(container)) {}

template <>
EntryObjectMDSPLUS::EntryObjectPlugin(const EntryObjectPlugin& other) : m_container_{other.m_container_} {}

template <>
EntryObjectMDSPLUS::EntryObjectPlugin(EntryObjectPlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
std::pair<std::shared_ptr<EntryObject>, Path> EntryObjectMDSPLUS::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<EntryObject>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const EntryObject>, Path> EntryObjectMDSPLUS::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const EntryObject>, Path>{nullptr, Path{}};
}

template <>
void EntryObjectMDSPLUS::load(const std::string& uri)
{
    VERBOSE << "Load MDSPLUS document :" << uri;
}

template <>
void EntryObjectMDSPLUS::save(const std::string& url) const
{
}

template <>
size_t EntryObjectMDSPLUS::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

template <>
void EntryObjectMDSPLUS::clear() { NOT_IMPLEMENTED; }

// template <>
// Entry EntryObjectMDSPLUS::at(const Path& path) { return Entry{entry_value_type{shared_from_this()}, path}; }
// template <>
// Entry EntryObjectMDSPLUS::at(const Path& path) const { return Entry{entry_value_type{const_cast<EntryObjectMDSPLUS*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const entry_value_type>
EntryObjectMDSPLUS::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

template <>
Cursor<entry_value_type>
EntryObjectMDSPLUS::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>{};
}

template <>
void EntryObjectMDSPLUS::for_each(std::function<void(const std::string&, const entry_value_type&)> const& visitor) const
{
}

template <>
entry_value_type EntryObjectMDSPLUS::insert(const std::string& key, entry_value_type v)
{
    entry_value_type res;
    NOT_IMPLEMENTED;
    return std::move(res);
}

template <>
entry_value_type EntryObjectMDSPLUS::find(const std::string& key) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectMDSPLUS::update(const std::string& key, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
void EntryObjectMDSPLUS::remove(const std::string& path) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectMDSPLUS::insert(const Path& path, entry_value_type v)
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectMDSPLUS::update(const Path& path, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectMDSPLUS::find(const Path& path) const
{
    std::ostringstream os;

    for (auto&& item : path)
    {
        switch (item.index())
        {
        case Path::segment_tags::Key:
            os << "/" << std::get<Path::segment_tags::Key>(item);
            break;
        case Path::segment_tags::Index:
            os << "[@id=" << std::get<Path::segment_tags::Index>(item) << "]";
            break;
        default:
            break;
        }
        // std::visit(
        //     sp::traits::overloaded{
        //         [&](std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) { os << "/" << key; },
        //         [&](std::variant_alternative_t<Path::segment_tags::Index, Path::Segment>& idx) { os << "[@id=" << idx << "]"; },
        //         [&](auto&&) {}},
        //     item);
    }

    VERBOSE << "XPath=" << os.str();
    entry_value_type res;
    return res;
}

template <>
void EntryObjectMDSPLUS::remove(const Path& path) { EntryObject::remove(path); }

SPDB_ENTRY_REGISTER(mdsplus, mdsplus_node);
SPDB_ENTRY_ASSOCIATE(mdsplus, mdsplus_node, "^(.*)\\.(mdsplus)$");

} // namespace sp::db