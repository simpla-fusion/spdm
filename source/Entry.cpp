#include "Entry.h"
#include "EntryInterface.h"
#include "utility/Factory.h"
#include "utility/Logger.h"
#include "utility/URL.h"
#include "utility/fancy_print.h"
#include <any>
#include <array>
#include <map>
#include <vector>
namespace sp
{
Entry::Entry(EntryInterface* p) : m_pimpl_(p != nullptr ? p : EntryInterface::create("memory").release())
{
    m_pimpl_->bind(this);
}

Entry::Entry(const std::string& uri) : m_pimpl_(nullptr) { fetch(uri); }

Entry::Entry(const this_type& other) : m_pimpl_(other.m_pimpl_->copy()) {}

Entry::Entry(this_type&& other) : m_pimpl_(other.m_pimpl_.release()) {}

Entry::~Entry() {}

void Entry::fetch(const std::string& uri)
{
    if (m_pimpl_ == nullptr)
    {
        std::string schema = "memory";

        auto pos = uri.find(":");

        if (pos == std::string::npos)
        {
            pos = uri.rfind('.');
            if (pos != std::string::npos)
            {
                schema = uri.substr(pos);
            }
            else
            {
                schema = uri;
            }
        }
        else
        {
            schema = uri.substr(0, pos);
        }

        if (schema == "")
        {
            schema = "memory";
        }
        else if (schema == "http" || schema == "https")
        {
            NOT_IMPLEMENTED;
        }
        if (!Factory<EntryInterface>::has_creator(schema))
        {
            RUNTIME_ERROR << "Can not parse schema " << schema << std::endl;
        }
        Factory<EntryInterface>::create(schema).swap(m_pimpl_);

        VERBOSE << "load backend:" << schema << std::endl;
        
        if (schema != uri)
        {
            m_pimpl_->fetch(uri);
        }
    }
    else
    {
        m_pimpl_->fetch(uri);
    }
}

void Entry::swap(this_type& other) { std::swap(m_pimpl_, other.m_pimpl_); }

Entry& Entry::operator=(this_type const& other)
{
    this_type(other).swap(*this);
    return *this;
}
void Entry::bind(Entry* parent, const std::string& name)
{
    m_parent_ = parent;
    m_name_ = name;
}

//
std::string Entry::prefix() const { return m_parent_ == nullptr ? m_name_ : (m_parent_->prefix() + "/" + m_name_); }

std::string Entry::name() const { return m_name_; }

// metadata
Entry::Type Entry::type() const { return m_pimpl_->type(); }
bool Entry::is_null() const { return type() == Type::Null; }
bool Entry::is_single() const { return type() == Type::Single; }
bool Entry::is_tensor() const { return type() == Type::Tensor; }
bool Entry::is_block() const { return type() == Type::Block; }
bool Entry::is_array() const { return type() == Type::Array; }
bool Entry::is_object() const { return type() == Type::Object; }

bool Entry::is_root() const { return m_parent_ == nullptr; }
bool Entry::is_leaf() const { return type() < Type::Array; };

// attributes
bool Entry::has_attribute(const std::string& name) const { return m_pimpl_->has_attribute(name); }

const Entry::single_t Entry::get_attribute_raw(const std::string& name) { return m_pimpl_->get_attribute_raw(name); }

void Entry::set_attribute_raw(const std::string& name, const single_t& value) { m_pimpl_->set_attribute_raw(name, value); }

void Entry::remove_attribute(const std::string& name) { m_pimpl_->remove_attribute(name); }

std::map<std::string, Entry::single_t> Entry::attributes() const { return m_pimpl_->attributes(); }

// as leaf
void Entry::set_single(const single_t& v) { m_pimpl_->set_single(v); }

Entry::single_t Entry::get_single() const { return m_pimpl_->get_single(); }

void Entry::set_tensor(const tensor_t& v) { m_pimpl_->set_tensor(v); }

Entry::tensor_t Entry::get_tensor() const { return m_pimpl_->get_tensor(); }

void Entry::set_block(const block_t& v) { m_pimpl_->set_block(v); }

Entry::block_t Entry::get_block() const { return m_pimpl_->get_block(); }

// as Tree
Entry::iterator Entry::parent() const { return Entry::iterator(m_parent_); }

Entry::const_iterator Entry::self() const { return this; }

Entry::iterator Entry::self() { return this; }

Entry::iterator Entry::next() const { return m_pimpl_->next(); }

// Entry::iterator Entry::first_child() const { return m_pimpl_->first_child(); }

// Entry::iterator Entry::last_child() const { return m_pimpl_->last_child(); }

Entry::range Entry::items() const { return m_pimpl_->items(); }

Range<Iterator<const std::pair<const std::string, Entry>>> Entry::children() const { return m_pimpl_->children(); };
// as container
size_t Entry::size() const { return m_pimpl_->size(); }

Entry::range Entry::find(const pred_fun& pred) { return m_pimpl_->find(pred); }

void Entry::erase(const iterator& p) { m_pimpl_->erase(p); }

void Entry::erase_if(const pred_fun& p) { m_pimpl_->erase_if(p); }

void Entry::erase_if(const range& r, const pred_fun& p) { m_pimpl_->erase_if(r, p); }

// as vector
Entry::iterator Entry::at(int idx) { return m_pimpl_->at(idx); }

Entry::iterator Entry::push_back() { return m_pimpl_->push_back(); }

Entry::iterator Entry::push_back(const Entry& other)
{
    auto p = push_back();
    Entry(other).swap(*p);
    return p;
}

Entry::iterator Entry::push_back(Entry&& other)
{
    auto p = push_back();
    p->swap(other);
    return p;
}

Entry Entry::pop_back() { NOT_IMPLEMENTED; }

Entry& Entry::operator[](int idx)
{
    if (idx < 0)
    {
        return *push_back();
    }
    else
    {
        auto p = at(idx);
        if (!p)
        {
            throw std::out_of_range(FILE_LINE_STAMP_STRING + "index out of range");
        }
        return *p;
    }
}

// as map
// @note : map is unordered
bool Entry::has_a(const std::string& name) { return !find(name); }

Entry::iterator Entry::find(const std::string& name) { return m_pimpl_->find(name); }

Entry::iterator Entry::at(const std::string& name)
{
    auto p = find(name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + name);
    }
    return p;
}

Entry& Entry::operator[](const std::string& name) { return *insert(name); }

Entry::iterator Entry::insert(const std::string& name) { return m_pimpl_->insert(name); }

Entry::iterator Entry::insert(const std::string& name, const Entry& other)
{
    auto p = insert(name);
    Entry(other).swap(*p);
    return p;
}

Entry::iterator Entry::insert(const std::string& name, Entry&& other)
{
    auto p = insert(name);
    p->swap(other);
    return p;
}

Entry Entry::erase(const std::string& name) { return m_pimpl_->erase(name); }

//-------------------------------------------------------------------
// level 2
size_t Entry::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

size_t Entry::height() const
{
    NOT_IMPLEMENTED;
    return 0;
}

Entry::range Entry::slibings() const { return range{next(), const_cast<this_type*>(this)->self()}; } // return slibings

Entry::range Entry::ancestor() const
{
    NOT_IMPLEMENTED;
    return range{};
}

Entry::range Entry::descendants() const
{
    NOT_IMPLEMENTED;
    return range{};
}

Entry::range Entry::leaves() const
{
    NOT_IMPLEMENTED;
    return range{};
}

Entry::range Entry::shortest_path(iterator const& target) const
{
    NOT_IMPLEMENTED;
    return range{};
}

ptrdiff_t Entry::distance(const this_type& target) const
{
    NOT_IMPLEMENTED;
    return 0;
}

std::ostream& fancy_print(std::ostream& os, const Entry& entry, int indent = 0)
{
    if (entry.type() == Entry::Type::Single)
    {
        auto v = entry.get_single();

        switch (v.index())
        {
        case 0:
            os << "\"" << std::get<std::string>(v) << "\"";
            break;
        case 1:
            os << std::get<bool>(v);
            break;
        case 2:
            os << std::get<int>(v);
            break;
        case 3:
            os << std::get<double>(v);
            break;
        case 4:
            os << std::get<std::complex<double>>(v);
            break;
        case 5:
        {
            auto d = std::get<std::array<int, 3>>(v);
            os << d[0] << "," << d[1] << "," << d[2];
        }
        break;
        case 6:
        {
            auto d = std::get<std::array<double, 3>>(v);
            os << d[0] << "," << d[1] << "," << d[2];
        }
        break;
        default:
            break;
        }
    }
    else if (entry.type() == Entry::Type::Array)
    {
        auto r = entry.items();
        os << "[ ";
        fancy_print_array1(os, r.first, r.second, indent);
        os << " ]";
    }
    else if (entry.type() == Entry::Type::Object)
    {
        auto r = entry.children();
        os << "{";
        fancy_print_key_value(os, r.first, r.second, indent, ":");
        os << "}";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, Entry const& entry)
{
    return fancy_print(os, entry, 0);
}

Entry load_entry_xml(const std::string& uri);

Entry load(const std::string& uri) { NOT_IMPLEMENTED; }

void save(const Entry&, const std::string& uri) { NOT_IMPLEMENTED; }

Entry load(const std::istream&, const std::string& format) { NOT_IMPLEMENTED; }

void save(const Entry&, const std::ostream&, const std::string& format) { NOT_IMPLEMENTED; }

} // namespace sp