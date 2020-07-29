
// #include "../EntryCursor.h"
#include "../EntryPlugin.h"
#include "../Node.h"
#include "../utility/Factory.h"
#include "../utility/HierarchicalNode.h"
#include "../utility/Logger.h"
#include <variant>
namespace sp
{

struct entry_memory : public HierarchicalNode
{
    entry_memory(const std::shared_ptr<Entry>& p = nullptr, const std::string& n = "")
        : Entry::type_union(nullptr), parent(p), name(n) {}
    ~entry_memory() = default;
    std::string name;
    std::shared_ptr<Entry> parent;
};

template <>
std::size_t EntryPlugin<entry_memory>::type() const { return m_pimpl_.index(); }
// template <>
// std::string EntryPlugin<entry_memory>::name() const { return ""; }
//----------------------------------------------------------------------------------
// level 0
template <>
std::string EntryPlugin<entry_memory>::name() const { return m_pimpl_.name; };

// as leaf
template <>
void EntryPlugin<entry_memory>::set_value(const Entry::type_union& v)
{
    if (type() >= Entry::type_tags::Array)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }

    Entry::type_union(v).swap(m_pimpl_);
}

template <>
Entry::type_union EntryPlugin<entry_memory>::get_value() const
{
    if (type() > Entry::type_tags::Array)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Element!");
    }
    return Entry::type_union(m_pimpl_);
}

// as Tree
// as object
template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::find(const std::string& name) const
{
    std::shared_ptr<Entry> res = nullptr;
    if (type() == Entry::type_tags::Object)
    {

        auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);
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
EntryPlugin<entry_memory>::find(const Path& xpath) const
{
    std::string path = xpath.str();
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

    if (type() == Entry::type_tags::Empty)
    {
        m_pimpl_.emplace<Entry::type_tags::Object>();
    }

    if (type() == Entry::type_tags::Object)
    {
        auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);
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
EntryPlugin<entry_memory>::insert(const Path& xpath)
{
    auto path = xpath.str();

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
void EntryPlugin<entry_memory>::erase(const std::string& name)
{
    if (type() == Entry::type_tags::Object)
    {
        auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);
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
    if (type() == Entry::type_tags::Object)
    {
        auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);
        res = m.size();
    }
    else if (type() == Entry::type_tags::Array)
    {
        auto& m = std::get<Entry::type_tags::Array>(m_pimpl_);
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
    if (type() == Entry::type_tags::Object)
    {
        auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);

        res = make_iterator<entry_memory>(m.begin(), m.end());
    }
    else if (type() == Entry::type_tags::Array)
    {
        auto& m = std::get<Entry::type_tags::Array>(m_pimpl_);

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
    if (type() == Entry::type_tags::Empty)
    {
        m_pimpl_.emplace<Entry::type_tags::Array>();
    }
    if (type() == Entry::type_tags::Array)
    {
        auto& m = std::get<Entry::type_tags::Array>(m_pimpl_);
        m.emplace_back(std::make_shared<this_type>(this->shared_from_this(), ""));
        res = *m.rbegin();
    }

    return res;
}

template <>
std::shared_ptr<Entry> EntryPlugin<entry_memory>::pop_back()
{
    std::shared_ptr<Entry> res = nullptr;

    if (type() == Entry::type_tags::Array)
    {
        auto& m = std::get<Entry::type_tags::Array>(m_pimpl_);
        res = *m.rbegin();
        m.pop_back();
    }

    return res;
}

template <>
std::shared_ptr<Entry>
EntryPlugin<entry_memory>::at(int idx) const
{
    std::shared_ptr<Entry> res = nullptr;

    if (type() == Entry::type_tags::Array)
    {
        auto& m = std::get<Entry::type_tags::Array>(m_pimpl_);
        res = m[idx];
    }

    return res;
}

// attributes
template <>
bool EntryPlugin<entry_memory>::has_attribute(const std::string& name) const { return find("@" + name) != nullptr; }

template <>
Entry::type_union EntryPlugin<entry_memory>::get_attribute_raw(const std::string& name) const
{
    auto p = find("@" + name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
    }
    return p->get_value();
}

template <>
void EntryPlugin<entry_memory>::set_attribute_raw(const std::string& name, const Entry::type_union& value)
{
    insert("@" + name)->set_element(value);
}

template <>
void EntryPlugin<entry_memory>::remove_attribute(const std::string& name) { remove("@" + name); }

template <>
std::map<std::string, Entry::type_union> EntryPlugin<entry_memory>::attributes() const
{
    if (type() != Entry::type_tags::Object)
    {
        return std::map<std::string, Entry::type_union>{};
    }

    std::map<std::string, Entry::type_union> res;
    for (const auto& item : std::get<Entry::type_tags::Object>(m_pimpl_))
    {
        if (item.first[0] == '@')
        {
            res.emplace(item.first.substr(1, std::string::npos), item.second->get_value());
        }
    }
    return std::move(res);
}

SP_REGISTER_ENTRY(memory, entry_memory);

} // namespace sp