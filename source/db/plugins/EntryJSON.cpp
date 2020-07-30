
#include "../Entry.h"
#include "../EntryInterface.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>
namespace sp::db
{
struct entry_json : public std::variant<nullptr_t,
                                       Entry::single_t,
                                       Entry::tensor_t,
                                       Entry::block_t,
                                       std::vector<Entry>,
                                       std::map<std::string, Entry>>
{
    typedef std::variant<nullptr_t,
                         Entry::single_t,
                         Entry::tensor_t,
                         Entry::block_t,
                         std::vector<Entry>,
                         std::map<std::string, Entry>>
        base_type;
    using base_type::variant;
};

template <>
EntryImplement<entry_json>::EntryImplement() : EntryInterface(), m_pimpl_( ){NOT_IMPLEMENTED;};

template <>
EntryImplement<entry_json>::EntryImplement(const EntryImplement& other) : EntryInterface(other), m_pimpl_(other.m_pimpl_) {}

template <>
EntryImplement<entry_json>::EntryImplement(EntryImplement&& other) : EntryInterface(std::forward<EntryImplement>(other)), m_pimpl_(std::move(other.m_pimpl_)) {}

template <>
EntryImplement<entry_json>::~EntryImplement() = default;
template <>
EntryInterface* EntryImplement<entry_json>::copy() const
{
    return new EntryImplement(*this);
};

template <>
EntryInterface* EntryImplement<entry_json>::duplicate() const { return new EntryImplement<entry_json>(); }

template <>
Entry::Type EntryImplement<entry_json>::type() const { return Entry::Type(m_pimpl_.index()); }

template <>
int EntryImplement<entry_json>::fetch(const std::string& uri)
{
    NOT_IMPLEMENTED;
}
//----------------------------------------------------------------------------------
// level 0
//
// as leaf
template <>
void EntryImplement<entry_json>::set_single(const Entry::single_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_pimpl_.emplace<Entry::Type::Single>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::single_t EntryImplement<entry_json>::get_single() const
{
    if (type() != Entry::Type::Single)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Single!");
    }
    return std::get<Entry::Type::Single>(m_pimpl_);
}
template <>
void EntryImplement<entry_json>::set_tensor(const Entry::tensor_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_pimpl_.emplace<Entry::Type::Tensor>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::tensor_t EntryImplement<entry_json>::get_tensor() const
{
    if (type() != Entry::Type::Tensor)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Tensor>(m_pimpl_);
}
template <>
void EntryImplement<entry_json>::set_block(const Entry::block_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_pimpl_.emplace<Entry::Type::Block>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::block_t EntryImplement<entry_json>::get_block() const
{
    if (type() != Entry::Type::Block)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Block>(m_pimpl_);
}

// as Tree

// as object
template <>
const Entry* EntryImplement<entry_json>::find(const std::string& name) const
{
    try
    {
        auto const& m = std::get<Entry::Type::Object>(m_pimpl_);
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
Entry* EntryImplement<entry_json>::find(const std::string& name)
{
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_pimpl_);
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
Entry* EntryImplement<entry_json>::insert(const std::string& name)
{
    if (type() == Entry::Type::Null)
    {
        m_pimpl_.emplace<Entry::Type::Object>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_pimpl_);

        return &(m.emplace(name, Entry(duplicate())).first->second);
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    }
}
template <>
Entry EntryImplement<entry_json>::erase(const std::string& name)
{
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_pimpl_);
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
Entry::iterator EntryImplement<entry_json>::next() const
{
    NOT_IMPLEMENTED;
    return Entry::iterator();
};
template <>
Range<Iterator<Entry>> EntryImplement<entry_json>::items() const
{
    if (type() == Entry::Type::Array)
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        return Entry::range{Entry::iterator(m.begin()),
                            Entry::iterator(m.end())};
        ;
    }
    // else if (type() == Entry::Type::Object)
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_pimpl_);
    //     auto mapper = [](auto const& item) -> Entry* { return &item->second; };
    //     return Entry::range{Entry::iterator(m.begin(), mapper),
    //                         Entry::iterator(m.end(), mapper)};
    // }

    return Entry::range{};
}
template <>
Range<Iterator<const std::pair<const std::string, Entry>>> EntryImplement<entry_json>::children() const
{
    if (type() == Entry::Type::Object)
    {
        auto& m = std::get<Entry::Type::Object>(m_pimpl_);

        return Range<Iterator<const std::pair<const std::string, Entry>>>{
            Iterator<const std::pair<const std::string, Entry>>(m.begin()),
            Iterator<const std::pair<const std::string, Entry>>(m.end())};
    }

    return Range<Iterator<const std::pair<const std::string, Entry>>>{};
}
template <>
size_t EntryImplement<entry_json>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Entry::range EntryImplement<entry_json>::find(const Entry::pred_fun& pred)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryImplement<entry_json>::erase(const Entry::iterator& p)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryImplement<entry_json>::erase_if(const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryImplement<entry_json>::erase_if(const Entry::range& r, const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}

// as vector
template <>
Entry* EntryImplement<entry_json>::at(int idx)
{
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        return &m[idx];
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    };
}
template <>
Entry* EntryImplement<entry_json>::push_back()
{
    if (type() == Entry::Type::Null)
    {
        m_pimpl_.emplace<Entry::Type::Array>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        m.emplace_back(Entry(duplicate()));
        return &*m.rbegin();
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    };
}
template <>
Entry EntryImplement<entry_json>::pop_back()
{
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
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
bool EntryImplement<entry_json>::has_attribute(const std::string& name) const { return !find("@" + name); }
template <>
Entry::single_t EntryImplement<entry_json>::get_attribute_raw(const std::string& name) const
{
    auto p = find("@" + name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
    }
    return p->get_single();
}
template <>
void EntryImplement<entry_json>::set_attribute_raw(const std::string& name, const Entry::single_t& value) { insert("@" + name)->set_single(value); }
template <>
void EntryImplement<entry_json>::remove_attribute(const std::string& name) { erase("@" + name); }
template <>
std::map<std::string, Entry::single_t> EntryImplement<entry_json>::attributes() const
{
    if (type() != Entry::Type::Object)
    {
        return std::map<std::string, Entry::single_t>{};
    }

    std::map<std::string, Entry::single_t> res;
    for (const auto& item : std::get<Entry::Type::Object>(m_pimpl_))
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