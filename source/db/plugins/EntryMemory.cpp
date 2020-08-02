
// #include "../EntryCursor.h"
#include "../../utility/Factory.h"
#include "../../utility/Logger.h"
#include "../EntryPlugin.h"
#include "../Node.h"
#include <variant>
namespace sp::db
{

typedef std::map<std::string, Entry::element> entry_memory;

template <>
struct cursor_traits<entry_memory>
{
    typedef entry_memory node_type;
    typedef node_type& reference;
    typedef node_type* pointer;
    typedef ptrdiff_t difference_type;
};
//----------------------------------------------------------------------------------------------------------
// as Hierarchy tree node
template <>
size_t EntryPlugin<entry_memory>::size() const { return m_pimpl_->size(); }

template <>
void EntryPlugin<entry_memory>::clear() { m_pimpl_->clear(); }

// function level 0

template <>
EntryPlugin<entry_memory>::const_cursor
EntryPlugin<entry_memory>::find(const std::string& name) const
{
    return make_cursor(m_pimpl_->find(name), m_pimpl_->end()).map<const element>();
};

template <>
EntryPlugin<entry_memory>::const_cursor
EntryPlugin<entry_memory>::find(const Path& xpath) const
{
    // std::string path = xpath.str();
    // int pos = 0;
    // auto res = const_cast<EntryPlugin<entry_memory>*>(this)->shared_from_this();

    // while (res != nullptr && pos < path.size())
    // {
    //     int end = path.find("/", pos);
    //     if (end == std::string::npos)
    //     {
    //         end = path.size();
    //     }
    //     res = res->find(path.substr(pos, end - pos));
    //     pos = end + 1;
    // }
    // return res;
    return find(xpath.str());
};

template <>
EntryPlugin<entry_memory>::cursor
EntryPlugin<entry_memory>::insert(const std::string& name)
{
    // Entry::cursor res = nullptr;

    // if (type() == Entry::type_tags::Empty)
    // {
    //     m_pimpl_.emplace<Entry::type_tags::Object>();
    // }

    // if (type() == Entry::type_tags::Object)
    // {
    //     auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);
    //     res = m.emplace(name, std::make_shared<this_type>(this->shared_from_this(), name)).first->second;
    // }
    // else
    // {
    //     throw std::runtime_error("Can not insert node to non-object!");
    // }

    return make_cursor(m_pimpl_->try_emplace(name).first, m_pimpl_->end()).map<element>();
}

template <>
EntryPlugin<entry_memory>::cursor
EntryPlugin<entry_memory>::insert(const Path& xpath)
{
    // auto path = xpath.str();

    // int pos = 0;
    // Entry::cursor res = shared_from_this();

    // while (res != nullptr && pos < path.size())
    // {
    //     int end = path.find("/", pos);
    //     if (end == std::string::npos)
    //     {
    //         end = path.size();
    //     }
    //     res = res->insert(path.substr(pos, end - pos));
    //     pos = end + 1;
    // }
    // return res;
    return insert(xpath.str());
}

template <>
void EntryPlugin<entry_memory>::erase(const std::string& name) { m_pimpl_->erase(m_pimpl_->find(name)); }

template <>
void EntryPlugin<entry_memory>::erase(const Path& xpath) { m_pimpl_->erase(m_pimpl_->find(xpath.str())); }

//----------------------------------------------------------------------------------------------------------
// child
template <>
std::size_t EntryPlugin<entry_memory>::count(const std::string& name) { return m_pimpl_->count(name); }

template <>
std::size_t
EntryPlugin<entry_memory>::type(const std::string& path) const
{
    auto cursor = find(path);
    return (!cursor.done()) ? type_tags::Empty : cursor->index();
};

// as leaf node,  need node.type

template <>
void EntryPlugin<entry_memory>::set_value(const std::string& path, const element& v)
{
    auto c = insert(path);
    if (c->index() > 2)
    {
        element(v).swap(*c);
    }
};

template <>
EntryPlugin<entry_memory>::element
EntryPlugin<entry_memory>::get_value(const std::string& path) const
{
    auto it = m_pimpl_->find(path);
    if (it == m_pimpl_->end())
    {
        throw std::out_of_range(path);
    }
    return *it;
};

//----------------------------------------------------------------------------------------------------------
// level 1
template <>
EntryPlugin<entry_memory>::cursor
EntryPlugin<entry_memory>::first_child() { return make_cursor(m_pimpl_->begin(), m_pimpl_->end()).map<element>(); }

template <>
EntryPlugin<entry_memory>::const_cursor
EntryPlugin<entry_memory>::first_child() const { return make_cursor(m_pimpl_->cbegin(), m_pimpl_->cend()).map<const element>(); }

template <>
EntryPlugin<entry_memory>::cursor
EntryPlugin<entry_memory>::select(const std::string& path) { return make_cursor(m_pimpl_->find(path), m_pimpl_->end()).map<element>(); }

template <>
EntryPlugin<entry_memory>::cursor
EntryPlugin<entry_memory>::select(const Path& path) { return select(path.str()); }

template <>
EntryPlugin<entry_memory>::const_cursor EntryPlugin<entry_memory>::select(const std::string& path) const
{
    return make_cursor(m_pimpl_->find(path), m_pimpl_->end()).map<const element>();
}

template <>
EntryPlugin<entry_memory>::const_cursor
EntryPlugin<entry_memory>::select(const Path& path) const { return select(path.str()); }

// template <>
// Entry::cursor
// EntryPlugin<entry_memory>::first_child() const
// {
//     Entry::cursor res{nullptr};
//     if (type() == Entry::type_tags::Object)
//     {
//         auto& m = std::get<Entry::type_tags::Object>(m_pimpl_);

//         res = make_iterator<entry_memory>(m.begin(), m.end());
//     }
//     else if (type() == Entry::2)
//     {
//         auto& m = std::get<Entry::2>(m_pimpl_);

//         res = make_iterator<entry_memory>(m.begin(), m.end());
//     }

//     return res;
// }
typedef std::vector<Entry::element> entry_memory_array;

// as array

template <>
Entry::cursor
EntryPluginArray<entry_memory_array>::push_back()
{
    m_pimpl_->emplace_back();
    return make_cursor(m_pimpl_->rbegin(), ++m_pimpl_->rbegin()).map<Entry::element>([](auto&& v) { return v.second; });
}

template <>
void EntryPluginArray<entry_memory_array>::pop_back() { m_pimpl_->pop_back(); }

template <>
const Entry::element&
EntryPluginArray<entry_memory_array>::at(int idx) const { return m_pimpl_->at(idx); }

template <>
Entry::element&
EntryPluginArray<entry_memory_array>::at(int idx) { return m_pimpl_->at(idx); }

//----------------------------------------------------------------------------------
// level 0

// as leaf

// // attributes
// template <>
// bool EntryPlugin<entry_memory>::has_attribute(const std::string& name) const { return find("@" + name) != nullptr; }

// template <>
// Entry::element EntryPlugin<entry_memory>::get_attribute_raw(const std::string& name) const
// {
//     auto p = find("@" + name);
//     if (!p)
//     {
//         throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
//     }
//     return p->get_value();
// }

// template <>
// void EntryPlugin<entry_memory>::set_attribute_raw(const std::string& name, const Entry::element& value)
// {
//     insert("@" + name)->set_element(value);
// }

// template <>
// void EntryPlugin<entry_memory>::remove_attribute(const std::string& name) { remove("@" + name); }

// template <>
// std::map<std::string, Entry::element> EntryPlugin<entry_memory>::attributes() const
// {
//     if (type() !=  type_tags::Object)
//     {
//         return std::map<std::string, Entry::element>{};
//     }

//     std::map<std::string, Entry::element> res;
//     for (const auto& item : std::get<Entry::type_tags::Object>(m_pimpl_))
//     {
//         if (item.first[0] == '@')
//         {
//             res.emplace(item.first.substr(1, std::string::npos), item.second->get_value());
//         }
//     }
//     return std::move(res);
// }

SPDB_REGISTER_ENTRY(memory, entry_memory);

} // namespace sp::db