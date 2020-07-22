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
template <typename... Others>
std::string join_path(const std::string& l, const std::string& r) { return (l == "") ? r : l + "/" + r; }

template <typename... Others>
std::string join_path(const std::string& l, const std::string& r, Others&&... others)
{
    return join_path(join_path(l, r), std::forward<Others>(others)...);
}

Entry::Entry()
    : m_prefix_(""), m_pimpl_(EntryInterface::create())
{
}

Entry::Entry(const std::string& uri)
    : m_prefix_(""), m_pimpl_(EntryInterface::create(uri)) {}

Entry::Entry(const std::shared_ptr<EntryInterface>& p, const std::string& prefix)
    : m_prefix_(prefix), m_pimpl_(p != nullptr ? p : EntryInterface::create()) {}

Entry::Entry(EntryInterface* p, const std::string& prefix)
    : Entry(p->shared_from_this(), prefix) {}

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

std::shared_ptr<EntryInterface> Entry::get_self() const
{
    auto p = m_prefix_ == "" ? m_pimpl_ : m_pimpl_->find_r(m_prefix_);

    if (p == nullptr)
    {
        throw std::out_of_range("Can not find node:" + m_prefix_);
    }
    
    return p;
}
std::shared_ptr<EntryInterface> Entry::get_self()
{
    if (m_prefix_ != "")
    {
        auto p = m_pimpl_->insert_r(m_prefix_);
        if (p == nullptr)
        {
            throw std::out_of_range("Can not find or insert node:" + m_prefix_);
        }
        m_pimpl_ = p;
        m_prefix_ = "";
    }
    return m_pimpl_;
}

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
bool Entry::has_attribute(const std::string& name) const { return get_self()->has_attribute(name); }

const Entry::single_t Entry::get_attribute_raw(const std::string& name) const { return get_self()->get_attribute_raw(name); }

void Entry::set_attribute_raw(const std::string& name, const single_t& value) { get_self()->set_attribute_raw(name, value); }

void Entry::remove_attribute(const std::string& name) { get_self()->remove_attribute(name); }

std::map<std::string, Entry::single_t> Entry::attributes() const { return get_self()->attributes(); }

// as leaf
void Entry::set_single(const single_t& v) { get_self()->set_single(v); }

Entry::single_t Entry::get_single() const { return get_self()->get_single(); }

void Entry::set_tensor(const tensor_t& v) { get_self()->set_tensor(v); }

Entry::tensor_t Entry::get_tensor() const { return get_self()->get_tensor(); }

void Entry::set_block(const block_t& v) { get_self()->set_block(v); }

Entry::block_t Entry::get_block() const { return get_self()->get_block(); }

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

Range<Entry> Entry::children() const
{
    return get_self()->children().map<Entry>([](const auto& item) { return Entry{item}; });
}

// as array

Entry Entry::push_back() { return Entry{get_self()->push_back()}; }

Entry Entry::pop_back() { return Entry{get_self()->pop_back()}; }

Entry Entry::operator[](int idx) { return Entry{get_self()->item(idx)}; }

Entry Entry::operator[](int idx) const
{

    return Entry{get_self()->item(idx)};
}

// as map
// @note : map is unordered
bool Entry::has_a(const std::string& name) const { return m_pimpl_->find(join_path(m_prefix_, name)) != nullptr; }

Entry Entry::find(const std::string& name) const { return Entry(m_pimpl_->find(join_path(m_prefix_, name))); }

Entry Entry::operator[](const std::string& name) { return Entry(m_pimpl_, join_path(m_prefix_, name)); }

Entry Entry::insert(const std::string& name) { return Entry(m_pimpl_->insert(join_path(m_prefix_, name))); }

void Entry::remove(const std::string& name) { m_pimpl_->remove(join_path(m_prefix_, name)); }

//-------------------------------------------------------------------
// level 2
size_t Entry::depth() const { return m_pimpl_ == nullptr ? 0 : parent().depth() + 1; }

size_t Entry::height() const
{
    NOT_IMPLEMENTED;
    return 0;
}

Range<Entry> Entry::slibings() const { return Range<Entry>{}; }

Range<Entry> Entry::ancestor() const
{
    NOT_IMPLEMENTED;
    return Range<Entry>{};
}

Range<Entry> Entry::descendants() const
{
    NOT_IMPLEMENTED;
    return Range<Entry>{};
}

Range<Entry> Entry::leaves() const
{
    NOT_IMPLEMENTED;
    return Range<Entry>{};
}

Range<Entry> Entry::shortest_path(Entry const& target) const
{
    NOT_IMPLEMENTED;
    return Range<Entry>{};
}

ptrdiff_t Entry::distance(const this_type& target) const
{
    NOT_IMPLEMENTED;
    return 0;
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
        fancy_print_array1(os, r.first, r.second, indent);
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

} // namespace sp