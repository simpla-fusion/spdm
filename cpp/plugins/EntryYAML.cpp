#include "../db/Entry.h"
#include "../db/EntryPlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>

namespace sp::db
{
struct yaml_node
{
};
typedef EntryObjectPlugin<yaml_node> EntryObjectYAML;

template <>
EntryObjectYAML::EntryObjectPlugin(const yaml_node& container) : m_container_(container) {}

template <>
EntryObjectYAML::EntryObjectPlugin(yaml_node&& container) : m_container_(std::move(container)) {}

template <>
EntryObjectYAML::EntryObjectPlugin(const EntryObjectPlugin& other) : m_container_{other.m_container_} {}

template <>
EntryObjectYAML::EntryObjectPlugin(EntryObjectPlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
std::pair<std::shared_ptr<EntryObject>, Path> EntryObjectYAML::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<EntryObject>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const EntryObject>, Path> EntryObjectYAML::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const EntryObject>, Path>{nullptr, Path{}};
}

template <>
void EntryObjectYAML::load(const std::string& uri)
{
    VERBOSE << "Load YAML document :" << uri;
}

template <>
void EntryObjectYAML::save(const std::string& url) const
{
}

template <>
size_t EntryObjectYAML::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

template <>
void EntryObjectYAML::clear() { NOT_IMPLEMENTED; }

// template <>
// Entry EntryObjectYAML::at(const Path& path) { return Entry{entry_value_type{shared_from_this()}, path}; }
// template <>
// Entry EntryObjectYAML::at(const Path& path) const { return Entry{entry_value_type{const_cast<EntryObjectYAML*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const entry_value_type>
EntryObjectYAML::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

template <>
Cursor<entry_value_type>
EntryObjectYAML::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>{};
}

template <>
void EntryObjectYAML::for_each(std::function<void(const std::string&, const entry_value_type&)> const& visitor) const
{
}

template <>
entry_value_type EntryObjectYAML::insert(const std::string& key, entry_value_type v)
{
    entry_value_type res;
    NOT_IMPLEMENTED;
    return std::move(res);
}

template <>
entry_value_type EntryObjectYAML::find(const std::string& key) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectYAML::update(const std::string& key, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
void EntryObjectYAML::remove(const std::string& path) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectYAML::insert(const Path& path, entry_value_type v)
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectYAML::update(const Path& path, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectYAML::find(const Path& path) const
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
void EntryObjectYAML::remove(const Path& path) { EntryObject::remove(path); }

SPDB_ENTRY_REGISTER(yaml, yaml_node);
SPDB_ENTRY_ASSOCIATE(yaml, yaml_node, "^(.*)\\.(yaml)$");

} // namespace sp::db