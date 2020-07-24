
#include "../EntryCursor.h"
#include "../EntryPlugin.h"
#include "../Node.h"
#include "../utility/Factory.h"
#include "../utility/Logger.h"
#include <variant>
namespace sp
{

typedef std::variant<std::nullptr_t,
                     Entry::element_t,
                     Entry::tensor_t,
                     Entry::block_t,
                     std::vector<std::shared_ptr<Entry>>,
                     std::map<std::string, std::shared_ptr<Entry>>>
    entry_memory_union;

struct entry_memory : public entry_memory_union
{
    entry_memory(const std::shared_ptr<Entry>& p = nullptr, const std::string& n = "") : entry_memory_union(nullptr), parent(p), name(n) {}
    ~entry_memory() = default;
    std::string name;
    std::shared_ptr<Entry> parent;
};

template <>
Entry::NodeType EntryPlugin<entry_memory>::type() const { return NodeType(m_pimpl_.index()); }
// template <>
// std::string EntryPlugin<entry_memory>::name() const { return ""; }
//----------------------------------------------------------------------------------
// level 0
template <>
std::string EntryPlugin<entry_memory>::name() const { return m_pimpl_.name; };

// as leaf
template <>
void EntryPlugin<entry_memory>::set_element(const Entry::element_t& v)
{
    if (type() >= NodeType::Array)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
    m_pimpl_.emplace<NodeType::Element>(v);
}

template <>
Entry::element_t EntryPlugin<entry_memory>::get_element() const
{
    if (type() != NodeType::Element)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Element!");
    }
    return std::get<NodeType::Element>(m_pimpl_);
}

template <>
void EntryPlugin<entry_memory>::set_tensor(const Entry::tensor_t& v)
{
    if (type() >= NodeType::Array)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
    m_pimpl_.emplace<NodeType::Tensor>(v);
}

template <>
Entry::tensor_t EntryPlugin<entry_memory>::get_tensor() const
{
    if (type() != NodeType::Tensor)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<NodeType::Tensor>(m_pimpl_);
}

template <>
void EntryPlugin<entry_memory>::set_block(const Entry::block_t& v)
{
    if (type() >= NodeType::Array)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
    m_pimpl_.emplace<NodeType::Block>(v);
}

template <>
Entry::block_t EntryPlugin<entry_memory>::get_block() const
{
    if (type() != NodeType::Block)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<NodeType::Block>(m_pimpl_);
}

// as Tree
// as object
template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::find(const std::string& name) const
{
    std::shared_ptr<Entry> res = nullptr;
    if (type() == NodeType::Object)
    {

        auto& m = std::get<NodeType::Object>(m_pimpl_);
        auto it = m.find(name);
        if (it != m.end())
        {
            res = it->second;
        }
    }

    return res;
};

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::find_r(const std::string& path) const
{
    int pos = 0;
    auto res = const_cast<EntryPlugin<entry_memory>*>(this)->shared_from_this();

    while (res != nullptr && pos < path.size())
    {
        int end = path.find("/", pos);
        if (end == std::string::npos)
        {
            end = path.size();
        }
        res = res->find(path.substr(pos, end - pos));
        pos = end + 1;
    }
    return res;
};

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::insert(const std::string& name)
{
    std::shared_ptr<Entry> res = nullptr;

    if (type() == NodeType::Null)
    {
        m_pimpl_.emplace<NodeType::Object>();
    }

    if (type() == NodeType::Object)
    {
        auto& m = std::get<NodeType::Object>(m_pimpl_);
        res = m.emplace(name, std::make_shared<this_type>(this->shared_from_this(), name)).first->second;
    }
    else
    {
        throw std::runtime_error("Can not insert node to non-object!");
    }

    return res;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::insert_r(const std::string& path)
{
    int pos = 0;
    std::shared_ptr<Entry> res = shared_from_this();

    while (res != nullptr && pos < path.size())
    {
        int end = path.find("/", pos);
        if (end == std::string::npos)
        {
            end = path.size();
        }
        res = res->insert(path.substr(pos, end - pos));
        pos = end + 1;
    }
    return res;
}

template <>
void EntryPlugin<entry_memory>::remove(const std::string& name)
{
    if (type() == NodeType::Object)
    {
        auto& m = std::get<NodeType::Object>(m_pimpl_);
        auto it = m.find(name);
        if (it != m.end())
        {
            m.erase(it);
        }
    }
}

template <>
size_t EntryPlugin<entry_memory>::size() const
{
    size_t res = 0;
    if (type() == NodeType::Object)
    {
        auto& m = std::get<NodeType::Object>(m_pimpl_);
        res = m.size();
    }
    else if (type() == NodeType::Array)
    {
        auto& m = std::get<NodeType::Array>(m_pimpl_);
        res = m.size();
    }
    return res;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::parent() const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::first_child() const
{
    std::shared_ptr<Entry> res{nullptr};
    if (type() == NodeType::Object)
    {
        auto& m = std::get<NodeType::Object>(m_pimpl_);

        res = make_iterator<entry_memory>(m.begin(), m.end());
    }
    else if (type() == NodeType::Array)
    {
        auto& m = std::get<NodeType::Array>(m_pimpl_);

        res = make_iterator<entry_memory>(m.begin(), m.end());
    }

    return res;
}

// as arraytemplate <>
template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::push_back()
{
    std::shared_ptr<Entry> res = nullptr;
    if (type() == NodeType::Null)
    {
        m_pimpl_.emplace<NodeType::Array>();
    }
    if (type() == NodeType::Array)
    {
        auto& m = std::get<NodeType::Array>(m_pimpl_);
        m.emplace_back(std::make_shared<this_type>(this->shared_from_this(), ""));
        res = *m.rbegin();
    }

    return res;
}

template <>
std::shared_ptr<Entry> EntryPlugin<entry_memory>::pop_back()
{
    std::shared_ptr<Entry> res = nullptr;

    if (type() == NodeType::Array)
    {
        auto& m = std::get<NodeType::Array>(m_pimpl_);
        res = *m.rbegin();
        m.pop_back();
    }

    return res;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::item(int idx) const
{
    std::shared_ptr<Entry> res = nullptr;

    if (type() == NodeType::Array)
    {
        auto& m = std::get<NodeType::Array>(m_pimpl_);
        res = m[idx];
    }

    return res;
}

// attributes
template <>
bool EntryPlugin<entry_memory>::has_attribute(const std::string& name) const { return find("@" + name) != nullptr; }

template <>
Entry::element_t EntryPlugin<entry_memory>::get_attribute_raw(const std::string& name) const
{
    auto p = find("@" + name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
    }
    return p->get_element();
}

template <>
void EntryPlugin<entry_memory>::set_attribute_raw(const std::string& name, const Entry::element_t& value)
{
    insert("@" + name)->set_element(value);
}

template <>
void EntryPlugin<entry_memory>::remove_attribute(const std::string& name) { remove("@" + name); }

template <>
std::map<std::string, Entry::element_t> EntryPlugin<entry_memory>::attributes() const
{
    if (type() != NodeType::Object)
    {
        return std::map<std::string, Entry::element_t>{};
    }

    std::map<std::string, Entry::element_t> res;
    for (const auto& item : std::get<NodeType::Object>(m_pimpl_))
    {
        if (item.first[0] == '@')
        {
            res.emplace(item.first.substr(1, std::string::npos), item.second->get_element());
        }
    }
    return std::move(res);
}

SP_REGISTER_ENTRY(memory, entry_memory);

} // namespace sp