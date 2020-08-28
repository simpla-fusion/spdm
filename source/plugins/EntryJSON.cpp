
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
EntryObjectJSON::EntryObjectPlugin(const EntryObjectPlugin& other) : m_container_(other.m_container_) {}

template <>
EntryObjectJSON::EntryObjectPlugin(EntryObjectPlugin&& other) : m_container_(std::move(other.m_container_)) {}
template <>
void EntryObjectJSON::for_each(std::function<void(const std::string&, const entry_value_type&)> const&) const {}

// access children
template <>
entry_value_type EntryObjectJSON::insert(const std::string&, entry_value_type) {}
template <>
entry_value_type EntryObjectJSON::find(const std::string& key) const {}

template <>
void EntryObjectJSON::update(const std::string& key, entry_value_type v) {}

template <>
void EntryObjectJSON::remove(const std::string& path) {}

template <>
entry_value_type EntryObjectJSON::insert(const Path& p ,entry_value_type) { return EntryObject::insert(std::move(v), path); }

template <>
void EntryObjectJSON::update(const Path& p ,entry_value_type) { EntryObject::update(std::move(v), path); }

template <>
entry_value_type EntryObjectJSON::find(const Path& path) const { return EntryObject::find(path); }

template <>
void EntryObjectJSON::remove(const Path& path) { EntryObject::remove(path); }

// as leaf
template <>
void EntryObjectJSON::set_single(const Entry::single_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_container_.emplace<Entry::Type::Single>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::single_t EntryObjectJSON::get_single() const
{
    if (type() != Entry::Type::Single)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Single!");
    }
    return std::get<Entry::Type::Single>(m_container_);
}
template <>
void EntryObjectJSON::set_tensor(const Entry::tensor_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_container_.emplace<Entry::Type::Tensor>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::tensor_t EntryObjectJSON::get_tensor() const
{
    if (type() != Entry::Type::Tensor)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Tensor>(m_container_);
}
template <>
void EntryObjectJSON::set_block(const Entry::block_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_container_.emplace<Entry::Type::Block>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::block_t EntryObjectJSON::get_block() const
{
    if (type() != Entry::Type::Block)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Block>(m_container_);
}

// as Tree

// as object
template <>
const Entry* EntryObjectJSON::find(const std::string& name) const
{
    try
    {
        auto const& m = std::get<Entry::Type::Object>(m_container_);
        auto it = m.find(name);
        if (it != m.end())
        {
            return &it->second;
        }
    }
    catch (std::bad_variant_access&)
    {
    }
    return nullptr;
}
template <>
Entry* EntryObjectJSON::find(const std::string& name)
{
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_container_);
        auto it = m.find(name);
        if (it != m.end())
        {
            return &it->second;
        }
    }
    catch (std::bad_variant_access&)
    {
    }
    return nullptr;
}
template <>
Entry* EntryObjectJSON::insert(const std::string& name)
{
    if (type() == Entry::Type::Null)
    {
        m_container_.emplace<Entry::Type::Object>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_container_);

        return &(m.emplace(name, Entry(duplicate())).first->second);
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    }
}
template <>
Entry EntryObjectJSON::erase(const std::string& name)
{
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_container_);
        auto it = m.find(name);
        if (it != m.end())
        {
            Entry res;
            res.swap(it->second);
            m.erase(it);
            return std::move(res);
        }
    }
    catch (std::bad_variant_access&)
    {
    }
    return Entry();
}

// Entry::iterator parent() const  { return Entry::iterator(const_cast<Entry*>(m_parent_)); }
template <>
Entry::iterator EntryObjectJSON::next() const
{
    NOT_IMPLEMENTED;
    return Entry::iterator();
};
template <>
Range<Iterator<Entry>> EntryObjectJSON::items() const
{
    if (type() == Entry::Type::Array)
    {
        auto& m = std::get<Entry::Type::Array>(m_container_);
        return Entry::range{Entry::iterator(m.begin()),
                            Entry::iterator(m.end())};
        ;
    }
    // else if (type() == Entry::Type::Object)
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_container_);
    //     auto mapper = [](auto const& item) -> Entry* { return &item->second; };
    //     return Entry::range{Entry::iterator(m.begin(), mapper),
    //                         Entry::iterator(m.end(), mapper)};
    // }

    return Entry::range{};
}
template <>
Range<Iterator<const std::pair<const std::string, Entry>>> EntryObjectJSON::children() const
{
    if (type() == Entry::Type::Object)
    {
        auto& m = std::get<Entry::Type::Object>(m_container_);

        return Range<Iterator<const std::pair<const std::string, Entry>>>{
            Iterator<const std::pair<const std::string, Entry>>(m.begin()),
            Iterator<const std::pair<const std::string, Entry>>(m.end())};
    }

    return Range<Iterator<const std::pair<const std::string, Entry>>>{};
}
template <>
size_t EntryObjectJSON::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Entry::range EntryObjectJSON::find(const Entry::pred_fun& pred)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryObjectJSON::erase(const Entry::iterator& p)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryObjectJSON::erase_if(const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryObjectJSON::erase_if(const Entry::range& r, const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}

// as vector
template <>
Entry* EntryObjectJSON::at(int idx)
{
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_container_);
        return &m[idx];
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    };
}
template <>
Entry* EntryObjectJSON::push_back()
{
    if (type() == Entry::Type::Null)
    {
        m_container_.emplace<Entry::Type::Array>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_container_);
        m.emplace_back(Entry(duplicate()));
        return &*m.rbegin();
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    };
}
template <>
Entry EntryObjectJSON::pop_back()
{
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_container_);
        Entry res;
        m.rbegin()->swap(res);
        m.pop_back();
        return std::move(res);
    }
    catch (std::bad_variant_access&)
    {
        return Entry();
    }
}

// attributes
template <>
bool EntryObjectJSON::has_attribute(const std::string& name) const { return !find("@" + name); }
template <>
Entry::single_t EntryObjectJSON::get_attribute_raw(const std::string& name) const
{
    auto p = find("@" + name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
    }
    return p->get_single();
}
template <>
void EntryObjectJSON::set_attribute_raw(const std::string& name, const Entry::single_t& value) { insert("@" + name)->set_single(value); }
template <>
void EntryObjectJSON::remove_attribute(const std::string& name) { erase("@" + name); }
template <>
std::map<std::string, Entry::single_t> EntryObjectJSON::attributes() const
{
    if (type() != Entry::Type::Object)
    {
        return std::map<std::string, Entry::single_t>{};
    }

    std::map<std::string, Entry::single_t> res;
    for (const auto& item : std::get<Entry::Type::Object>(m_container_))
    {
        if (item.first[0] == '@')
        {
            res.emplace(item.first.substr(1, std::string::npos), item.second.get_single());
        }
    }
    return std::move(res);
}

SP_REGISTER_ENTRY(json, entry_json);

} // namespace sp::db