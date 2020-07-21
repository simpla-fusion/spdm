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
Entry::Entry() : m_pimpl_(nullptr), m_parent_(nullptr), m_name_("") {}

Entry::Entry(const std::string& uri) : m_pimpl_(EntryInterface::create(uri)), m_parent_(nullptr), m_name_("") {}

Entry::Entry(Entry* parent, const std::string& name) : m_pimpl_(nullptr), m_parent_(parent), m_name_(name) {}

Entry::Entry(const std::shared_ptr<EntryInterface>& p) : m_pimpl_(p), m_parent_(nullptr), m_name_("") {}

Entry::Entry(const Entry& other) : m_pimpl_(other.m_pimpl_), m_name_(other.m_name_), m_parent_(other.m_parent_) {}

Entry::Entry(Entry&& other) : m_pimpl_(other.m_pimpl_), m_name_(other.m_name_), m_parent_(other.m_parent_) { other.m_pimpl_.reset(); }

Entry::~Entry() {}

std::shared_ptr<EntryInterface> Entry::get(const std::string& path)
{

    std::shared_ptr<EntryInterface> res = m_pimpl_;

    if (m_pimpl_ != nullptr)
    {
        res = path == "" ? m_pimpl_ : m_pimpl_->find(path);
    }
    else if (m_parent_ == nullptr)
    {
        res = EntryInterface::create();
    }
    else if (path == "")
    {
        m_pimpl_ = m_parent_->get(m_name_);
        res = m_pimpl_;
    }
    else
    {
        res = m_parent_->get(m_name_ + "." + path);
    }
    return res;
}
std::shared_ptr<EntryInterface> Entry::get(const std::string& path) const
{

    std::shared_ptr<EntryInterface> res = m_pimpl_;

    if (m_pimpl_ != nullptr)
    {
        res = path == "" ? m_pimpl_ : m_pimpl_->find(path);
    }
    else if (m_parent_ == nullptr)
    {
        res = nullptr;
    }
    else if (path == "")
    {
        res = m_pimpl_;
    }
    else
    {
        res = m_parent_->get(m_name_ + "." + path);
    }
    return res;
}
Entry Entry::fetch(const std::string& uri)
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
        m_pimpl_ = Factory<EntryInterface>::create(schema);

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

    return Entry{};
}

void Entry::swap(this_type& other)
{
    std::swap(m_name_, other.m_name_);
    std::swap(m_parent_, other.m_parent_);
    std::swap(m_pimpl_, other.m_pimpl_);
}

Entry& Entry::operator=(this_type const& other)
{
    this_type(other).swap(*this);
    return *this;
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
// as container

Entry Entry::parent() const { return m_parent_ == nullptr ? Entry() : Entry(*m_parent_); }

Entry const& Entry::self() const { return *this; }

Entry& Entry::self() { return *this; }

Entry::range Entry::children() const
{
    NOT_IMPLEMENTED;
    return Entry::range{};
};

void Entry::remove(const Entry& p) { m_pimpl_->remove(p.name()); }

// as array

Entry Entry::push_back()
{
    Entry res(m_pimpl_->push_back());
    res.m_parent_ = this;
    return std::move(res);
}

Entry Entry::pop_back() { return Entry{m_pimpl_->pop_back()}; }

Entry Entry::operator[](int idx)
{
    if (idx < 0)
    {
        return push_back();
    }
    else
    {
        Entry res(m_pimpl_->item(idx));

        if (res.m_pimpl_ == nullptr)
        {
            throw std::out_of_range(FILE_LINE_STAMP_STRING + "index out of range");
        }
        res.m_parent_ = this;
        return std::move(res);
    }
}

// as map
// @note : map is unordered
bool Entry::has_a(const std::string& name) const { return get()->find(name) != nullptr; }

Entry Entry::find(const std::string& name) const { return Entry(m_pimpl_->find(name)); }

Entry Entry::operator[](const std::string& name) { return Entry(this, name); }

Entry Entry::insert(const std::string& name)
{
    Entry res(m_pimpl_->insert(name));
    return std::move(res);
}

void Entry::remove(const std::string& name) { m_pimpl_->remove(name); }

//-------------------------------------------------------------------
// level 2
size_t Entry::depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

size_t Entry::height() const
{
    NOT_IMPLEMENTED;
    return 0;
}

Entry::range Entry::slibings() const { return range{}; }

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

Entry::range Entry::shortest_path(Entry const& target) const
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
        auto r = entry.children();
        os << "[ ";
        // fancy_print_array1(os, r.first, r.second, indent);
        os << " ]";
    }
    else if (entry.type() == Entry::Type::Object)
    {
        auto r = entry.children();
        os << "{";
        // fancy_print_key_value(os, r.first, r.second, indent, ":");
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

std::string to_string(Entry::single_t const& s)
{
    std::ostringstream os;
    switch (s.index())
    {
    case 0: // std::string
        os << std::get<0>(s);
        break;
    case 1: // bool
        os << std::boolalpha << std::get<1>(s);
        break;
    case 2: // int
        os << std::get<2>(s);
        break;
    case 3: // double
        os << std::get<3>(s);
        break;
    case 4: // std::complex<4>
        os << std::get<4>(s);
        break;
    case 5: //   std::array<int, 3>,
        os << std::get<5>(s)[0] << "," << std::get<5>(s)[1] << "," << std::get<5>(s)[2];
        break;
    case 6: //   std::array<int, 3>,
        os << std::get<6>(s)[0] << "," << std::get<6>(s)[1] << "," << std::get<6>(s)[2];
        break;

    default:
        break;
    }
    return os.str();
}

Entry::single_t from_string(const std::string& s)
{
    NOT_IMPLEMENTED;
}
} // namespace sp