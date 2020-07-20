
#include "../Entry.h"
#include "../EntryInterface.h"
#include "../utility/Logger.h"
#include "pugixml/pugixml.hpp"
#include <variant>
namespace sp
{

typedef pugi::xml_node entry_xml;

template <>
EntryImplement<entry_xml>::EntryImplement() : EntryInterface(), m_pimpl_(){};

template <>
EntryImplement<entry_xml>::EntryImplement(const entry_xml& other) : EntryInterface(), m_pimpl_(other){};

template <>
EntryImplement<entry_xml>::EntryImplement(const EntryImplement& other) : EntryInterface(other), m_pimpl_(other.m_pimpl_) {}

template <>
EntryImplement<entry_xml>::EntryImplement(EntryImplement&& other) : EntryInterface(std::forward<EntryImplement>(other)), m_pimpl_(std::move(other.m_pimpl_)) {}

template <>
EntryImplement<entry_xml>::~EntryImplement() = default;

template <>
EntryInterface* EntryImplement<entry_xml>::copy() const { return new EntryImplement(*this); };

template <>
EntryInterface* EntryImplement<entry_xml>::duplicate() const { return new EntryImplement<entry_xml>(); }

template <>
Entry::Type EntryImplement<entry_xml>::type() const { return Entry::Type(0); }

template <>
int EntryImplement<entry_xml>::fetch(const std::string& uri)
{
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(uri.c_str(), pugi::parse_default | pugi::parse_pi);
    if (result)
    {
        VERBOSE << uri << std::endl;

        m_pimpl_ = doc.first_child();
    }
    else
    {
        VERBOSE << "Can not load" << uri << std::endl;
    }
    return 0;
}
//----------------------------------------------------------------------------------
// level 0
//
// as leaf
template <>
void EntryImplement<entry_xml>::set_single(const Entry::single_t& v)
{
    if (type() < Entry::Type::Array)
    {
        // m_pimpl_.emplace<Entry::Type::Single>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
    NOT_IMPLEMENTED;
}
template <>
Entry::single_t EntryImplement<entry_xml>::get_single() const
{
    if (type() != Entry::Type::Single)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Single!");
    }
    NOT_IMPLEMENTED;
    return Entry::single_t{nullptr};
}
template <>
void EntryImplement<entry_xml>::set_tensor(const Entry::tensor_t& v)
{
    if (type() < Entry::Type::Array)
    {
        // m_pimpl_.emplace<Entry::Type::Tensor>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::tensor_t EntryImplement<entry_xml>::get_tensor() const
{
    if (type() != Entry::Type::Tensor)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }

    NOT_IMPLEMENTED;
    return Entry::tensor_t{nullptr, typeid(nullptr_t), {}};
}
template <>
void EntryImplement<entry_xml>::set_block(const Entry::block_t& v)
{
    if (type() < Entry::Type::Array)
    {
        NOT_IMPLEMENTED;

        // m_pimpl_.emplace<Entry::Type::Block>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}
template <>
Entry::block_t EntryImplement<entry_xml>::get_block() const
{
    if (type() != Entry::Type::Block)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    NOT_IMPLEMENTED;
    return Entry::block_t{nullptr, (nullptr), {}, {}, {}, {}};
}

// as Tree

// as object
template <>
const Entry* EntryImplement<entry_xml>::find(const std::string& name) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry* EntryImplement<entry_xml>::find(const std::string& name)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry* EntryImplement<entry_xml>::insert(const std::string& name)
{

    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry EntryImplement<entry_xml>::erase(const std::string& name)
{
    // try
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_pimpl_);
    //     auto it = m.find(name);
    //     if (it != m.end())
    //     {
    //         Entry res;
    //         res.swap(it->second);
    //         m.erase(it);
    //         return std::move(res);
    //     }
    // }
    // catch (std::bad_variant_access&)
    // {
    // }
    NOT_IMPLEMENTED;
    return Entry();
}

// Entry::iterator parent() const  { return Entry::iterator(const_cast<Entry*>(m_parent_)); }
template <>
Entry::iterator EntryImplement<entry_xml>::next() const
{
    NOT_IMPLEMENTED;
    return Entry::iterator();
};
template <>
Range<Iterator<Entry>> EntryImplement<entry_xml>::items() const
{
    if (type() == Entry::Type::Array)
    {
        // auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        // return Entry::range{Entry::iterator(m.begin()),
        //                     Entry::iterator(m.end())};
        // ;
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
Range<Iterator<const std::pair<const std::string, Entry>>> EntryImplement<entry_xml>::children() const
{
    // if (type() == Entry::Type::Object)
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_pimpl_);

    //     return Range<Iterator<const std::pair<const std::string, Entry>>>{
    //         Iterator<const std::pair<const std::string, Entry>>(m.begin()),
    //         Iterator<const std::pair<const std::string, Entry>>(m.end())};
    // }

    return Range<Iterator<const std::pair<const std::string, Entry>>>{};
}
template <>
size_t EntryImplement<entry_xml>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Entry::range EntryImplement<entry_xml>::find(const Entry::pred_fun& pred)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryImplement<entry_xml>::erase(const Entry::iterator& p)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryImplement<entry_xml>::erase_if(const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryImplement<entry_xml>::erase_if(const Entry::range& r, const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}

// as vector
template <>
Entry* EntryImplement<entry_xml>::at(int idx)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry* EntryImplement<entry_xml>::push_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry EntryImplement<entry_xml>::pop_back()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

// attributes
template <>
bool EntryImplement<entry_xml>::has_attribute(const std::string& name) const { return !find("@" + name); }
template <>
Entry::single_t EntryImplement<entry_xml>::get_attribute_raw(const std::string& name) const
{
    NOT_IMPLEMENTED;
    return Entry::single_t{nullptr};
}
template <>
void EntryImplement<entry_xml>::set_attribute_raw(const std::string& name, const Entry::single_t& value) { insert("@" + name)->set_single(value); }
template <>
void EntryImplement<entry_xml>::remove_attribute(const std::string& name) { erase("@" + name); }
template <>
std::map<std::string, Entry::single_t> EntryImplement<entry_xml>::attributes() const
{
    // if (type() != Entry::Type::Object)
    // {
    //     return std::map<std::string, Entry::single_t>{};
    // }

    std::map<std::string, Entry::single_t> res;
    // for (const auto& item : std::get<Entry::Type::Object>(m_pimpl_))
    // {
    //     if (item.first[0] == '@')
    //     {
    //         res.emplace(item.first.substr(1, std::string::npos), item.second.get_single());
    //     }
    // }
    NOT_IMPLEMENTED;
    return std::move(res);
}

SP_REGISTER_ENTRY(xml, entry_xml);

} // namespace sp