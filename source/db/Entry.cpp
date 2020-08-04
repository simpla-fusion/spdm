#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
namespace sp::db
{
class EntryArrayDefault;
class EntryObjectDefault;
//-----------------------------------------------------------------------------------------------------------
Entry::Entry() {}

Entry::~Entry() {}

Entry& Entry::fetch() { return base_type::index() == type_tags::Reference ? *std::get<type_tags::Reference>(*this) : *this; }

const Entry& Entry::fetch() const { return base_type::index() == type_tags::Reference ? *std::get<type_tags::Reference>(*this) : *this; }

void Entry::update()
{
    if (base_type::index() == type_tags::Reference)
    {
        std::get<type_tags::Reference>(*this)->update();
    }
}

std::size_t Entry::type() const { return fetch().index(); }

void Entry::clear() { base_type::emplace<std::nullptr_t>(nullptr); }

std::shared_ptr<DataBlock> Entry::as_block()
{
    switch (base_type::index())
    {
    case type_tags::Empty:
        emplace<type_tags::Block>(std::make_shared<DataBlock>());
        break;
    case type_tags::Block:
        break;
    case type_tags::Reference:
        std::get<type_tags::Reference>(*this)->as_block();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Block>(fetch());
}

std::shared_ptr<const DataBlock> Entry::as_block() const
{

    if (type() != type_tags::Block)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Block>(fetch());
}

std::shared_ptr<EntryObject> Entry::as_object()
{
    switch (base_type::index())
    {
    case type_tags::Empty:
        emplace<type_tags::Object>(EntryObject::create(this));
        break;
    case type_tags::Object:
        break;
    case type_tags::Reference:
        std::get<type_tags::Reference>(*this)->as_object();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Object>(fetch());
}

std::shared_ptr<const EntryObject> Entry::as_object() const
{
    if (type() != type_tags::Object)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Object>(fetch());
}

std::shared_ptr<EntryArray> Entry::as_array()
{
    switch (base_type::index())
    {
    case type_tags::Empty:
        emplace<type_tags::Array>(EntryArray::create(this));
        break;
    case type_tags::Array:
        break;

    case type_tags::Reference:
        std::get<type_tags::Reference>(*this)->as_array();
        update();
        break;
    default:
        throw std::runtime_error("illegal type");
        break;
    }
    return std::get<type_tags::Array>(fetch());
}

std::shared_ptr<const EntryArray> Entry::as_array() const
{
    if (index() != type_tags::Array)
    {
        throw std::runtime_error("illegal type");
    }
    return std::get<type_tags::Array>(fetch());
}

//==========================================================================================
EntryObject::EntryObject(Entry* s) : m_self_(s) {}

EntryObject::~EntryObject() {}

std::shared_ptr<EntryObject> EntryObject::create(Entry* self, const std::string& request)
{
    if (request == "")
    {
        return std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryObjectDefault>(self));
    }

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
        obj = std::dynamic_pointer_cast<EntryObject>(std::make_shared<EntryObjectDefault>(self));
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
    obj->self(self);
    return obj;
}

bool EntryObject::add_creator(const std::string& c_id, const std::function<EntryObject*()>& fun)
{
    return Factory<EntryObject>::add(c_id, fun);
};

//==========================================================================================

EntryArray::EntryArray(Entry* s) : m_self_(s) {}

EntryArray::~EntryArray() {}

std::shared_ptr<EntryArray> EntryArray::create(Entry* self, const std::string& request)
{
    auto res = std::dynamic_pointer_cast<EntryArray>(std::make_shared<EntryArrayDefault>(self));
    res->self(self);
    return res;
};

//==========================================================================================

class EntryObjectDefault : public EntryObject
{
    std::map<std::string, std::shared_ptr<Entry>> m_container_;

public:
    typedef EntryObjectDefault this_type;
    typedef Entry::type_tags type_tags;

    EntryObjectDefault(Entry* self) : EntryObject(self) {}

    EntryObjectDefault(const this_type& other) : EntryObject(nullptr), m_container_(other.m_container_) {}

    EntryObjectDefault(EntryObjectDefault&& other) : EntryObject(nullptr), m_container_(std::move(other.m_container_)) {}

    ~EntryObjectDefault() = default;

    std::shared_ptr<EntryObject> copy() const override { return std::shared_ptr<EntryObject>(new this_type(*this)); }

    //----------------------------------------------------------------------------------------------------------

    size_t size() const override;

    void clear() override;

    //------------------------------------------------------------------

    Cursor<Entry> select(const XPath& path) override;

    Cursor<const Entry> select(const XPath& path) const override;

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> kv_items() override;

    Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> kv_items() const override;

    //------------------------------------------------------------------

    std::shared_ptr<Entry> insert(const std::string& path) override;

    std::shared_ptr<Entry> insert(const XPath& path) override;

    std::shared_ptr<const Entry> get(const std::string& path) const override;

    std::shared_ptr<const Entry> get(const XPath& path) const override;

    void erase(const std::string& path) override;

    void erase(const XPath& path) override;
};

size_t EntryObjectDefault::size() const { return m_container_.size(); }

void EntryObjectDefault::clear() { m_container_.clear(); }

std::shared_ptr<Entry>
EntryObjectDefault::insert(const std::string& name)
{
    auto res = m_container_.try_emplace(name);
    if (res.second)
    {
        res.first->second.reset(new Entry);
    }
    return res.first->second;
}

std::shared_ptr<Entry>
EntryObjectDefault::insert(const XPath& path)
{

    Entry* p = self();
    for (auto it = path.begin(); it != path.end(); ++it)
    {
        switch (it->index())
        {
        case XPath::type_tags::Key:
            p = p->as_object()->insert(std::get<XPath::type_tags::Key>(*it)).get();
            break;
        case XPath::type_tags::Index:
            p = p->as_array()->get(std::get<XPath::type_tags::Index>(*it)).get();
            break;
        default:
            NOT_IMPLEMENTED;
            break;
        }
    }
    return p->shared_from_this();
}

std::shared_ptr<const Entry>
EntryObjectDefault::get(const std::string& path) const { return m_container_.at(path); }

std::shared_ptr<const Entry>
EntryObjectDefault::get(const XPath& path) const
{

    const Entry* p = self();
    for (auto it = path.begin(); it != path.end(); ++it)
    {
        switch (it->index())
        {
        case XPath::type_tags::Key:
            p = p->as_object()->get(std::get<XPath::type_tags::Key>(*it)).get();
            break;
        case XPath::type_tags::Index:
            p = p->as_array()->get(std::get<XPath::type_tags::Index>(*it)).get();
            break;
        default:
            NOT_IMPLEMENTED;
            break;
        }
    }
    return p->shared_from_this();
}

void EntryObjectDefault::erase(const std::string& path) { m_container_.erase(m_container_.find(path)); }

void EntryObjectDefault::erase(const XPath& path) { NOT_IMPLEMENTED; }

//--------------------------------------------------------------------------------------------------------------------------------------------

Cursor<Entry>
EntryObjectDefault::select(const XPath& path)
{
    NOT_IMPLEMENTED;
    return nullptr;
}

Cursor<const Entry>
EntryObjectDefault::select(const XPath& path) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

Cursor<Entry>
EntryObjectDefault::children()
{
    return make_cursor(m_container_.begin(), m_container_.end())
        .map<Entry>([](const std::pair<const std::string&, std::shared_ptr<Entry>>& item) -> Entry& { return *item.second; });
}

Cursor<const Entry>
EntryObjectDefault::children() const
{
    return make_cursor(m_container_.cbegin(), m_container_.cend())
        .map<const Entry>([](const std::pair<const std::string&, std::shared_ptr<Entry>>& item) -> const Entry& { return *item.second; });
}

Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> EntryObjectDefault::kv_items()
{
    return make_cursor(m_container_.begin(), m_container_.end());
};

Cursor<std::pair<const std::string, std::shared_ptr<Entry>>> EntryObjectDefault::kv_items() const
{
    return make_cursor(m_container_.cbegin(), m_container_.cend());
};

//--------------------------------------------------------------------------------

class EntryArrayDefault : public EntryArray
{
private:
    std::vector<std::shared_ptr<Entry>> m_container_;

public:
    typedef Entry::type_tags type_tags;
    typedef EntryArrayDefault this_type;

    EntryArrayDefault(Entry* self) : EntryArray(self) {}

    ~EntryArrayDefault() = default;

    EntryArrayDefault(const this_type& other) : EntryArray(nullptr), m_container_(other.m_container_) {}

    EntryArrayDefault(EntryArrayDefault&& other) : EntryArray(nullptr), m_container_(std::move(other.m_container_)) {}
    //--------------------------------------------------------------------------------------
    // as array

    std::shared_ptr<EntryArray> copy() const override { return std::dynamic_pointer_cast<EntryArray>(std::make_shared<EntryArrayDefault>(*this)); }

    size_t size() const override;

    void resize(std::size_t num) override;

    void clear() override;

    Cursor<Entry> children() override;

    Cursor<const Entry> children() const override;

    //--------------------------------------------------------------------------------------
    std::shared_ptr<Entry> push_back() override;

    void pop_back() override;

    std::shared_ptr<Entry> get(int idx) override;

    std::shared_ptr<const Entry> get(int idx) const override;
};

size_t EntryArrayDefault::size() const { return m_container_.size(); }

void EntryArrayDefault::resize(std::size_t num)
{

    auto s = m_container_.size();
    m_container_.resize(num);
    for (int i = s; i < num; ++i)
    {
        if (m_container_[i] == nullptr)
        {
            m_container_[i].reset(new Entry);
        }
    }
}

void EntryArrayDefault::clear() { m_container_.clear(); }

Cursor<Entry> EntryArrayDefault::children()
{
    return make_cursor(m_container_.begin(), m_container_.end())
        .map<Entry>([](const std::shared_ptr<Entry>& item) -> Entry& { return *item; });
}

Cursor<const Entry> EntryArrayDefault::children() const
{
    return make_cursor(m_container_.begin(), m_container_.end())
        .map<const Entry>([](const std::shared_ptr<Entry>& item) -> const Entry& { return *item; });
}

//--------------------------------------------------------------------------------------

std::shared_ptr<Entry> EntryArrayDefault::push_back()
{
    auto& p = m_container_.emplace_back();
    if (p == nullptr)
    {
        p.reset(new Entry);
    }
    return p;
}

void EntryArrayDefault::pop_back() { m_container_.pop_back(); }

std::shared_ptr<const Entry> EntryArrayDefault::get(int idx) const { return m_container_.at(idx); }

std::shared_ptr<Entry> EntryArrayDefault::get(int idx) { return m_container_.at(idx); }

} // namespace sp::db
namespace sp::utility
{
std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    std::visit(sp::traits::overloaded{
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Array, sp::db::Entry::base_type>& ele) {
                       os << "[";
                       for (auto it = ele->children(); !it.done(); it.next())
                       {
                           os << std::endl
                              << std::setw(indent * tab) << " ";
                           fancy_print(os, *it, indent + 1, tab);
                           os << ",";
                       }
                       os << std::endl
                          << std::setw(indent * tab)
                          << "]";
                   },
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Object, sp::db::Entry::base_type>& ele) {
                       os << "{";
                       for (auto it = ele->kv_items(); !it.done(); it.next())
                       {
                           os << std::endl
                              << std::setw(indent * tab) << " "
                              << "\"" << it->first << "\" : ";
                           fancy_print(os, *(it->second), indent + 1, tab);
                           os << ",";
                       }
                       os << std::endl
                          << std::setw(indent * tab)
                          << "}";
                   },
                   [&](const std::variant_alternative_t<sp::db::Entry::type_tags::Empty, sp::db::Entry::base_type>& ele) { fancy_print(os, nullptr, indent + 1, tab); },
                   [&](auto&& ele) { fancy_print(os, ele, indent + 1, tab); } //
               },
               dynamic_cast<const sp::db::Entry::base_type&>(entry));

    // if (entry.type() == Entry::NodeType::Element)
    // {
    //     os << to_string(entry.get_element());
    // }
    // else if (entry.type() == Entry::NodeType::Array)
    // else if (entry.type() == Entry::NodeType::Object)
    // {
    //
    // }
    return os;
}
} // namespace sp::utility
namespace sp::db
{
std::ostream& operator<<(std::ostream& os, Entry const& entry) { return sp::utility::fancy_print(os, entry, 0); }
} // namespace sp::db