#include "../db/Entry.h"
#include "../db/EntryPlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>

namespace sp::db
{
struct json_node
{
};
typedef EntryObjectPlugin<json_node> EntryObjectJSON;

template <>
EntryObjectJSON::EntryObjectPlugin(const json_node& container) : m_container_(container) {}

template <>
EntryObjectJSON::EntryObjectPlugin(json_node&& container) : m_container_(std::move(container)) {}

template <>
EntryObjectJSON::EntryObjectPlugin(const EntryObjectPlugin& other) : m_container_{other.m_container_} {}

template <>
EntryObjectJSON::EntryObjectPlugin(EntryObjectPlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
std::pair<std::shared_ptr<EntryObject>, Path> EntryObjectJSON::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<EntryObject>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const EntryObject>, Path> EntryObjectJSON::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const EntryObject>, Path>{nullptr, Path{}};
}

template <>
void EntryObjectJSON::load(const std::string& uri)
{
    VERBOSE << "Load JSON document :" << uri;
}

template <>
void EntryObjectJSON::save(const std::string& url) const
{
}

template <>
size_t EntryObjectJSON::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

template <>
void EntryObjectJSON::clear() { NOT_IMPLEMENTED; }

// template <>
// Entry EntryObjectJSON::at(const Path& path) { return Entry{entry_value_type{shared_from_this()}, path}; }
// template <>
// Entry EntryObjectJSON::at(const Path& path) const { return Entry{entry_value_type{const_cast<EntryObjectJSON*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const entry_value_type>
EntryObjectJSON::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

template <>
Cursor<entry_value_type>
EntryObjectJSON::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>{};
}

template <>
void EntryObjectJSON::for_each(std::function<void(const std::string&, const entry_value_type&)> const& visitor) const
{
}

template <>
entry_value_type EntryObjectJSON::insert(const std::string& key, entry_value_type v)
{
    entry_value_type res;
    NOT_IMPLEMENTED;
    return std::move(res);
}

template <>
entry_value_type EntryObjectJSON::find(const std::string& key) const
{
    NOT_IMPLEMENTED;
    OUT_OF_RANGE << key;
    return entry_value_type{};
}

template <>
void EntryObjectJSON::update(const std::string& key, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
void EntryObjectJSON::remove(const std::string& path) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectJSON::insert(const Path& path, entry_value_type v)
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectJSON::update(const Path& path, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectJSON::find(const Path& path) const
{
    OUT_OF_RANGE << path.str();
    throw std::out_of_range("Missing key:" + path.str());

    // std::ostringstream os;

    // for (auto&& item : path)
    // {
    //     switch (item.index())
    //     {
    //     case Path::segment_tags::Key:
    //         os << "/" << std::get<Path::segment_tags::Key>(item);
    //         break;
    //     case Path::segment_tags::Index:
    //         os << "[@id=" << std::get<Path::segment_tags::Index>(item) << "]";
    //         break;
    //     default:
    //         break;
    //     }
    //     // std::visit(
    //     //     sp::traits::overloaded{
    //     //         [&](std::variant_alternative_t<Path::segment_tags::Key, Path::Segment>& key) { os << "/" << key; },
    //     //         [&](std::variant_alternative_t<Path::segment_tags::Index, Path::Segment>& idx) { os << "[@id=" << idx << "]"; },
    //     //         [&](auto&&) {}},
    //     //     item);
    // }

    // VERBOSE << "XPath=" << os.str();
    entry_value_type res;
    return res;
}

template <>
void EntryObjectJSON::remove(const Path& path) { EntryObject::remove(path); }

SPDB_ENTRY_REGISTER(json, json_node);
SPDB_ENTRY_ASSOCIATE(json, json_node, "^(.*)\\.(json)$");

} // namespace sp::db