#include "../db/Entry.h"
#include "../db/EntryPlugin.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>

namespace sp::db
{
struct hdf5_node
{
};
typedef EntryObjectPlugin<hdf5_node> EntryObjectHDF5;

template <>
EntryObjectHDF5::EntryObjectPlugin(const hdf5_node& container) : m_container_(container) {}

template <>
EntryObjectHDF5::EntryObjectPlugin(hdf5_node&& container) : m_container_(std::move(container)) {}

template <>
EntryObjectHDF5::EntryObjectPlugin(const EntryObjectPlugin& other) : m_container_{other.m_container_} {}

template <>
EntryObjectHDF5::EntryObjectPlugin(EntryObjectPlugin&& other) : m_container_{std::move(other.m_container_)} {}

template <>
std::pair<std::shared_ptr<EntryObject>, Path> EntryObjectHDF5::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<EntryObject>, Path>{nullptr, Path{}};
}

template <>
std::pair<std::shared_ptr<const EntryObject>, Path> EntryObjectHDF5::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const EntryObject>, Path>{nullptr, Path{}};
}

template <>
void EntryObjectHDF5::load(const std::string& uri)
{
    VERBOSE << "Load HDF5 document :" << uri;
}

template <>
void EntryObjectHDF5::save(const std::string& url) const
{
}

template <>
size_t EntryObjectHDF5::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

template <>
void EntryObjectHDF5::clear() { NOT_IMPLEMENTED; }

// template <>
// Entry EntryObjectHDF5::at(const Path& path) { return Entry{entry_value_type{shared_from_this()}, path}; }
// template <>
// Entry EntryObjectHDF5::at(const Path& path) const { return Entry{entry_value_type{const_cast<EntryObjectHDF5*>(this)->shared_from_this()}, path}; }

template <>
Cursor<const entry_value_type>
EntryObjectHDF5::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const entry_value_type>{};
}

template <>
Cursor<entry_value_type>
EntryObjectHDF5::children()
{
    NOT_IMPLEMENTED;
    return Cursor<entry_value_type>{};
}

template <>
void EntryObjectHDF5::for_each(std::function<void(const std::string&, const entry_value_type&)> const& visitor) const
{
}

template <>
entry_value_type EntryObjectHDF5::insert(const std::string& key, entry_value_type v)
{
    entry_value_type res;
    NOT_IMPLEMENTED;
    return std::move(res);
}

template <>
entry_value_type EntryObjectHDF5::find(const std::string& key) const
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectHDF5::update(const std::string& key, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
void EntryObjectHDF5::remove(const std::string& path) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectHDF5::insert(const Path& path, entry_value_type v)
{
    NOT_IMPLEMENTED;
    return entry_value_type{};
}

template <>
void EntryObjectHDF5::update(const Path& path, entry_value_type v) { NOT_IMPLEMENTED; }

template <>
entry_value_type EntryObjectHDF5::find(const Path& path) const
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
void EntryObjectHDF5::remove(const Path& path) { EntryObject::remove(path); }

SPDB_ENTRY_REGISTER(hdf5, hdf5_node);
SPDB_ENTRY_ASSOCIATE(hdf5, hdf5_node, "^(.*)\\.(hdf5|h5)$");

} // namespace sp::db