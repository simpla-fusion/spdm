#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "EntryPlugin.h"
namespace sp::db
{

EntryObject& Entry::as_object()
{
    switch (index())
    {
    case type_tags::Empty:
        emplace<type_tags::Object>(EntryObject::create());
        break;
    case type_tags::Object:
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return *std::get<type_tags::Object>(*this);
}

const EntryObject& Entry::as_object() const
{
    if (index() != type_tags::Object)
    {
        throw std::runtime_error("illegal type");
    }
    return *std::get<type_tags::Object>(*this);
}

EntryArray& Entry::as_array()
{
    switch (index())
    {
    case type_tags::Empty:
        emplace<type_tags::Array>(EntryArray::create());
        break;
    case type_tags::Array:
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return *std::get<type_tags::Array>(*this);
}

const EntryArray& Entry::as_array() const
{
    if (index() != type_tags::Array)
    {
        throw std::runtime_error("illegal type");
    }
    return *std::get<type_tags::Array>(*this);
}

//-----------------------------------------------------------------------------------------------------------

EntryObject::~EntryObject() {}

std::shared_ptr<EntryObject> EntryObject::create(const std::string& request)
{

    std::string schema = "";

    auto pos = request.find(":");

    if (pos == std::string::npos)
    {
        pos = request.rfind('.');
        if (pos != std::string::npos)
        {
            schema = request.substr(pos);
        }
        else
        {
            schema = request;
        }
    }
    else
    {
        schema = request.substr(0, pos);
    }

    if (schema == "http" || schema == "https")
    {
        NOT_IMPLEMENTED;
    }

    std::shared_ptr<EntryObject> obj;

    if (schema == "")
    {
        obj = std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryPluginObject<std::map<std::string, Entry>>>());
    }
    else if (Factory<EntryObject>::has_creator(schema))
    {
        obj = std::shared_ptr<EntryObject>(Factory<EntryObject>::create(schema).release());
    }
    else
    {
        RUNTIME_ERROR << "Can not parse schema " << schema << std::endl;
    }

    if (obj == nullptr)
    {
        throw std::runtime_error("Can not create Entry for schema: " + schema);
    }
    else
    {
        VERBOSE << "load backend:" << schema << std::endl;
    }

    // if (schema != request)
    // {
    //     res->fetch(request);
    // }

    return obj;
}

bool EntryObject::add_creator(const std::string& c_id, const std::function<EntryObject*()>& fun)
{
    return Factory<EntryObject>::add(c_id, fun);
};

//-----------------------------------------------------------------------------------------------------------

EntryArray::~EntryArray() {}
std::shared_ptr<EntryArray> EntryArray::create(const std::string& request)
{
    return std::dynamic_pointer_cast<EntryArray>(std::make_shared<EntryPluginArray<std::vector<Entry>>>());
};

//-----------------------------------------------------------------------------------------------------------

typedef std::map<std::string, Entry> entry_memory;

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
size_t EntryPluginObject<entry_memory>::size() const { return m_pimpl_->size(); }

template <>
void EntryPluginObject<entry_memory>::clear() { m_pimpl_->clear(); }

// function level 0

template <>
Cursor<const Entry>
EntryPluginObject<entry_memory>::find(const std::string& name) const
{
    return make_cursor(m_pimpl_->find(name), m_pimpl_->end()).map<const Entry>();
};

template <>
Cursor<const Entry>
EntryPluginObject<entry_memory>::find(const Path& xpath) const
{
    // std::string path = xpath.str();
    // int pos = 0;
    // auto res = const_cast<EntryPluginObject<entry_memory>*>(this)->shared_from_this();

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
Cursor<Entry>
EntryPluginObject<entry_memory>::insert(const std::string& name)
{
    return make_cursor(m_pimpl_->try_emplace(name).first, m_pimpl_->end()).map<Entry>();
}

template <>
Cursor<Entry>
EntryPluginObject<entry_memory>::insert(const Path& xpath)
{
    // auto path = xpath.str();

    // int pos = 0;
    // Cursor<Entry> res = shared_from_this();

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
void EntryPluginObject<entry_memory>::erase(const std::string& name) { m_pimpl_->erase(m_pimpl_->find(name)); }

template <>
void EntryPluginObject<entry_memory>::erase(const Path& xpath) { m_pimpl_->erase(m_pimpl_->find(xpath.str())); }

//----------------------------------------------------------------------------------------------------------
// child
template <>
std::size_t EntryPluginObject<entry_memory>::count(const std::string& name) { return m_pimpl_->count(name); }

//----------------------------------------------------------------------------------------------------------
// level 1
template <>
Cursor<Entry>
EntryPluginObject<entry_memory>::first_child() { return make_cursor(m_pimpl_->begin(), m_pimpl_->end()).map<Entry>(); }

template <>
Cursor<const Entry>
EntryPluginObject<entry_memory>::first_child() const { return make_cursor(m_pimpl_->cbegin(), m_pimpl_->cend()).map<const Entry>(); }

template <>
Cursor<Entry>
EntryPluginObject<entry_memory>::find(const std::string& path) { return make_cursor(m_pimpl_->find(path), m_pimpl_->end()).map<Entry>(); }

template <>
Cursor<Entry>
EntryPluginObject<entry_memory>::find(const Path& path) { return find(path.str()); }

template <>
Cursor<Entry>
EntryPluginObject<entry_memory>::select(const std::string& path) { return make_cursor(m_pimpl_->find(path), m_pimpl_->end()).map<Entry>(); }

template <>
Cursor<Entry>
EntryPluginObject<entry_memory>::select(const Path& path) { return select(path.str()); }

template <>
Cursor<const Entry> EntryPluginObject<entry_memory>::select(const std::string& path) const
{
    return make_cursor(m_pimpl_->find(path), m_pimpl_->end()).map<const Entry>();
}

template <>
Cursor<const Entry>
EntryPluginObject<entry_memory>::select(const Path& path) const { return select(path.str()); }

// template <>
// Cursor<Entry>
// EntryPluginObject<entry_memory>::first_child() const
// {
//     Cursor<Entry> res{nullptr};
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
typedef std::vector<Entry> entry_memory_array;

// as array
template <>
std::shared_ptr<EntryArray> EntryPluginArray<entry_memory_array>::copy() const
{
    auto res = std::make_shared<EntryPluginArray<entry_memory_array>>();
    std::vector<Entry>(*m_pimpl_).swap(*res->m_pimpl_);
    return std::dynamic_pointer_cast<EntryArray>(res);
};

template <>
size_t EntryPluginArray<entry_memory_array>::size() const { return m_pimpl_->size(); }

template <>
void EntryPluginArray<entry_memory_array>::resize(std::size_t num) { m_pimpl_->resize(num); }

template <>
void EntryPluginArray<entry_memory_array>::clear() { m_pimpl_->clear(); }

template <>
Cursor<Entry>
EntryPluginArray<entry_memory_array>::push_back()
{
    m_pimpl_->emplace_back();
    return make_cursor(&*m_pimpl_->rbegin()); //.map<Entry>([](auto&& v) -> Entry& { return v.second; });
}

template <>
void EntryPluginArray<entry_memory_array>::pop_back() { m_pimpl_->pop_back(); }

template <>
const Entry&
EntryPluginArray<entry_memory_array>::at(int idx) const { return m_pimpl_->at(idx); }

template <>
Entry&
EntryPluginArray<entry_memory_array>::at(int idx) { return m_pimpl_->at(idx); }

} // namespace sp::db