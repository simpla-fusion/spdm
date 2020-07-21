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
Entry::Entry()
    : m_prefix_(""), m_pimpl_(EntryInterface::create()) {}

Entry::Entry(const std::string& uri)
    : m_prefix_(""), m_pimpl_(EntryInterface::create(uri)) {}

Entry::Entry(const std::shared_ptr<EntryInterface>& p, const std::string& prefix)
    : m_prefix_(prefix), m_pimpl_(p != nullptr ? p : EntryInterface::create()) {}

Entry::Entry(const Entry& other)
    : m_prefix_(other.m_prefix_), m_pimpl_(other.m_pimpl_) {}

Entry::Entry(Entry&& other)
    : m_prefix_(other.m_prefix_), m_pimpl_(other.m_pimpl_) { other.m_pimpl_.reset(); }

Entry::~Entry() {}

void Entry::swap(this_type& other)
{
    std::swap(m_prefix_, other.m_prefix_);
    std::swap(m_pimpl_, other.m_pimpl_);
}

Entry& Entry::operator=(this_type const& other)
{
    this_type(other).swap(*this);
    return *this;
}

//
std::string Entry::full_path() const
{
    NOT_IMPLEMENTED;
    return "/" + m_prefix_;
}

std::string Entry::relative_path() const { return m_prefix_; }

// metadata
Entry::Type Entry::type() const { return m_pimpl_->type(); }
bool Entry::is_null() const { return type() == Type::Null; }
bool Entry::is_single() const { return type() == Type::Single; }
bool Entry::is_tensor() const { return type() == Type::Tensor; }
bool Entry::is_block() const { return type() == Type::Block; }
bool Entry::is_array() const { return type() == Type::Array; }
bool Entry::is_object() const { return type() == Type::Object; }

bool Entry::is_root() const { return m_pimpl_->parent() == nullptr; }
bool Entry::is_leaf() const { return type() < Type::Array; };

// attributes
bool Entry::has_attribute(const std::string& name) const { return m_pimpl_->find(m_prefix_)->has_attribute(name); }

const Entry::single_t Entry::get_attribute_raw(const std::string& name) { return m_pimpl_->find(m_prefix_)->get_attribute_raw(name); }

void Entry::set_attribute_raw(const std::string& name, const single_t& value) { m_pimpl_->find(m_prefix_)->set_attribute_raw(name, value); }

void Entry::remove_attribute(const std::string& name) { m_pimpl_->find(m_prefix_)->remove_attribute(name); }

std::map<std::string, Entry::single_t> Entry::attributes() const { return m_pimpl_->find(m_prefix_)->attributes(); }

// as leaf
void Entry::set_single(const single_t& v) { m_pimpl_->insert(m_prefix_)->set_single(v); }

Entry::single_t Entry::get_single() const { return m_pimpl_->find(m_prefix_)->get_single(); }

void Entry::set_tensor(const tensor_t& v) { m_pimpl_->insert(m_prefix_)->set_tensor(v); }

Entry::tensor_t Entry::get_tensor() const { return m_pimpl_->find(m_prefix_)->get_tensor(); }

void Entry::set_block(const block_t& v) { m_pimpl_->insert(m_prefix_)->set_block(v); }

Entry::block_t Entry::get_block() const { return m_pimpl_->find(m_prefix_)->get_block(); }

// as Tree
// as container

Entry Entry::parent() const
{
    auto pos = m_prefix_.rfind("/");
    if (pos == std::string::npos)
    {
        return Entry(m_pimpl_->parent());
    }
    else
    {
        return Entry(m_pimpl_->parent(), m_prefix_.substr(0, pos));
    }
}

Entry const& Entry::self() const { return *this; }

Entry& Entry::self() { return *this; }

Entry::range Entry::children() const
{
    return m_pimpl_
        ->find(m_prefix_)
        ->children()
        .template map<Entry>([](const std::string& k, const std::shared_ptr<EntryInterface>& p) -> Entry { return Entry{p}; });
}

// as array

Entry Entry::push_back() { return Entry{m_pimpl_->insert(m_prefix_)->push_back()}; }

Entry Entry::pop_back() { return Entry{m_pimpl_->insert(m_prefix_)->pop_back()}; }

Entry Entry::operator[](int idx) { return Entry{m_pimpl_->insert(m_prefix_)->item(idx)}; }

// as map
// @note : map is unordered
bool Entry::has_a(const std::string& name) const { return m_pimpl_->find(m_prefix_ + "/" + name) != nullptr; }

Entry Entry::find(const std::string& name) const { return Entry(m_pimpl_->find(m_prefix_ + "/" + name)); }

Entry Entry::operator[](const std::string& name) { return Entry(m_pimpl_, m_prefix_ + "/" + name); }

Entry Entry::insert(const std::string& name) { return Entry(m_pimpl_->insert(m_prefix_ + "/" + name)); }

void Entry::remove(const std::string& name) { m_pimpl_->remove(m_prefix_ + "/" + name); }

//-------------------------------------------------------------------
// level 2
size_t Entry::depth() const { return m_pimpl_ == nullptr ? 0 : parent().depth() + 1; }

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