
#include "../Entry.h"
#include "../EntryInterface.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>
namespace sp
{
typedef std::variant<nullptr_t,
                     Entry::single_t,
                     Entry::tensor_t,
                     Entry::block_t,
                     std::vector<std::shared_ptr<EntryInterface>>,
                     std::map<std::string, std::shared_ptr<EntryInterface>>>
    entry_memory;

template <>
EntryImplement<entry_memory>::EntryImplement()
    : EntryInterface(),
      m_pimpl_(nullptr){};

template <>
EntryImplement<entry_memory>::EntryImplement(const entry_memory& other)
    : EntryInterface(),
      m_pimpl_(other){};

template <>
EntryImplement<entry_memory>::EntryImplement(const EntryImplement& other)
    : EntryInterface(other),
      m_pimpl_(other.m_pimpl_) {}

template <>
EntryImplement<entry_memory>::EntryImplement(EntryImplement&& other)
    : EntryInterface(std::forward<EntryImplement>(other)),
      m_pimpl_(std::move(other.m_pimpl_)) {}

template <>
EntryImplement<entry_memory>::~EntryImplement() = default;

template <>
std::shared_ptr<EntryInterface> EntryImplement<entry_memory>::copy() const
{
    return std::make_shared<EntryImplement<entry_memory>>(*this);
};

template <>
std::shared_ptr<EntryInterface> EntryImplement<entry_memory>::duplicate() const
{
    return std::make_shared<EntryImplement<entry_memory>>();
}

template <>
Entry::Type EntryImplement<entry_memory>::type() const { return Entry::Type(m_pimpl_.index()); }

// template <>
// int EntryImplement<entry_memory>::fetch(const std::string& uri)
// {
//     NOT_IMPLEMENTED;
// }
//----------------------------------------------------------------------------------
// level 0
//
// as leaf
template <>
void EntryImplement<entry_memory>::set_single(const Entry::single_t& v)
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
Entry::single_t EntryImplement<entry_memory>::get_single() const
{
    if (type() != Entry::Type::Single)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Single!");
    }
    return std::get<Entry::Type::Single>(m_pimpl_);
}

template <>
void EntryImplement<entry_memory>::set_tensor(const Entry::tensor_t& v)
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
Entry::tensor_t EntryImplement<entry_memory>::get_tensor() const
{
    if (type() != Entry::Type::Tensor)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Tensor>(m_pimpl_);
}

template <>
void EntryImplement<entry_memory>::set_block(const Entry::block_t& v)
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
Entry::block_t EntryImplement<entry_memory>::get_block() const
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
std::shared_ptr<EntryInterface>
EntryImplement<entry_memory>::find(const std::string& name) const
{
    return nullptr;
};

template <>
std::shared_ptr<EntryInterface>
EntryImplement<entry_memory>::insert(const std::string& name)
{
    if (type() == Entry::Type::Null)
    {
        m_pimpl_.emplace<Entry::Type::Object>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_pimpl_);

        return m.emplace(name, duplicate()).first->second;
    }
    catch (std::bad_variant_access&)
    {
        return nullptr;
    }
}

template <>
void EntryImplement<entry_memory>::remove(const std::string& name)
{
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_pimpl_);
        auto it = m.find(name);
        if (it != m.end())
        {
            m.erase(it);
        }
    }
    catch (std::bad_variant_access&)
    {
    }
}

template <>
size_t EntryImplement<entry_memory>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

template <>
std::shared_ptr<EntryInterface>
EntryImplement<entry_memory>::parent() const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
Range<EntryInterface>
EntryImplement<entry_memory>::children() const
{
    // if (type() == Entry::Type::Object)
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_pimpl_);

    //     return Range<Iterator<const std::pair<const std::string, std::shared_ptr<Entry>>>>{
    //         Iterator<const std::pair<const std::string, std::shared_ptr<Entry>>>(m.begin()),
    //         Iterator<const std::pair<const std::string, std::shared_ptr<Entry>>>(m.end())};
    // }

    return Range<EntryInterface>{};
}

// as arraytemplate <>
template <>
std::shared_ptr<EntryInterface> EntryImplement<entry_memory>::push_back()
{
    if (type() == Entry::Type::Null)
    {
        m_pimpl_.emplace<Entry::Type::Array>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        m.emplace_back(duplicate());
        return *m.rbegin();
    }
    catch (std::bad_variant_access&)
    {
    };

    return nullptr;
}

template <>
std::shared_ptr<EntryInterface> EntryImplement<entry_memory>::pop_back()
{
    // try
    // {
    //     auto& m = std::get<Entry::Type::Array>(m_pimpl_);
    //     std::shared_ptr<EntryInterface> res = *m.rbegin();
    //     m.pop_back();
    //     return res;
    // }
    // catch (std::bad_variant_access&)
    // {

    // }

    return nullptr;
}

template <>
std::shared_ptr<EntryInterface> EntryImplement<entry_memory>::item(int idx) const
{
    std::shared_ptr<EntryInterface> res = nullptr;
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        res = m[idx];
    }
    catch (std::bad_variant_access&)
    {
        RUNTIME_ERROR << "illegal entry type!" << std::endl;
    };
    return res;
}

template <>
Range<EntryInterface>
EntryImplement<entry_memory>::items() const
{
    Range<EntryInterface> res{};

    if (type() == Entry::Type::Array)
    {
        auto& m = std::get<Entry::Type::Array>(m_pimpl_);
        auto mapper = [](auto const& p) -> EntryInterface* { return p->get(); };

        Range<EntryInterface>{m.begin(), m.end(), mapper}.swap(res);
    }
    // else if (type() == Entry::Type::Object)
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_pimpl_);
    //     auto mapper = [](auto const& item) -> EntryInterface* { return item->second; };
    //     //  Range<EntryInterface>{m.begin(), m.end(), mapper}.swap(res);
    // }

    return res;
}

// attributes
template <>
bool EntryImplement<entry_memory>::has_attribute(const std::string& name) const { return find("@" + name) != nullptr; }

template <>
Entry::single_t EntryImplement<entry_memory>::get_attribute_raw(const std::string& name) const
{
    auto p = find("@" + name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
    }
    return p->get_single();
}
template <>

void EntryImplement<entry_memory>::set_attribute_raw(const std::string& name, const Entry::single_t& value)
{
    insert("@" + name)->set_single(value);
}

template <>
void EntryImplement<entry_memory>::remove_attribute(const std::string& name) { remove("@" + name); }

template <>
std::map<std::string, Entry::single_t> EntryImplement<entry_memory>::attributes() const
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
            res.emplace(item.first.substr(1, std::string::npos), item.second->get_single());
        }
    }
    return std::move(res);
}

SP_REGISTER_ENTRY(memory, entry_memory);

} // namespace sp