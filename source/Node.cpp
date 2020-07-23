#include "Node.h"
#include "Entry.h"
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

Node::Node()
    : m_prefix_(""), m_entry_(Entry::create())
{
}

Node::Node(const std::string& uri)
    : m_prefix_(""), m_entry_(Entry::create(uri)) {}

Node::Node(const std::shared_ptr<Entry>& p, const std::string& prefix)
    : m_prefix_(prefix), m_entry_(p != nullptr ? p : Entry::create()) {}

Node::Node(Entry* p, const std::string& prefix)
    : Node(p->shared_from_this(), prefix) {}

Node::Node(const Node& other)
    : m_prefix_(other.m_prefix_), m_entry_(other.m_entry_) {}

Node::Node(Node&& other)
    : m_prefix_(other.m_prefix_), m_entry_(other.m_entry_) { other.m_entry_.reset(); }

Node::~Node() {}

void Node::swap(this_type& other)
{
    std::swap(m_prefix_, other.m_prefix_);
    std::swap(m_entry_, other.m_entry_);
}

Node& Node::operator=(this_type const& other)
{
    this_type(other).swap(*this);
    return *this;
}

//

std::shared_ptr<Entry> Node::get_entry() const
{
    auto p = m_prefix_ == "" ? m_entry_ : m_entry_->find_r(m_prefix_);

    if (p == nullptr)
    {
        throw std::out_of_range("Can not find node:" + m_prefix_);
    }

    return p;
}

std::shared_ptr<Entry> Node::get_entry()
{
    if (m_prefix_ != "")
    {
        auto p = m_entry_->insert_r(m_prefix_);
        if (p == nullptr)
        {
            throw std::out_of_range("Can not find or insert node:" + m_prefix_);
        }
        m_entry_ = p;
        m_prefix_ = "";
    }
    return m_entry_;
}

std::string Node::full_path() const
{
    NOT_IMPLEMENTED;
    return "/" + m_prefix_;
}

std::string Node::relative_path() const { return m_prefix_; }

// metadata
Entry::Type Node::type() const { return m_entry_->type(); }
bool Node::is_null() const { return type() == Entry::Type::Null; }
bool Node::is_element() const { return type() == Entry::Type::Element; }
bool Node::is_tensor() const { return type() == Entry::Type::Tensor; }
bool Node::is_block() const { return type() == Entry::Type::Block; }
bool Node::is_array() const { return type() == Entry::Type::Array; }
bool Node::is_object() const { return type() == Entry::Type::Object; }

bool Node::is_root() const { return m_entry_->parent() == nullptr; }
bool Node::is_leaf() const { return type() < Entry::Type::Array; };

// attributes
bool Node::has_attribute(const std::string& name) const { return get_entry()->has_attribute(name); }

const Entry::element_t Node::get_attribute_raw(const std::string& name) const { return get_entry()->get_attribute_raw(name); }

void Node::set_attribute_raw(const std::string& name, const Entry::element_t& value) { get_entry()->set_attribute_raw(name, value); }

void Node::remove_attribute(const std::string& name) { get_entry()->remove_attribute(name); }

std::map<std::string, Entry::element_t> Node::attributes() const { return get_entry()->attributes(); }

// as leaf
void Node::set_element(const Entry::element_t& v) { get_entry()->set_element(v); }

Entry::element_t Node::get_element() const { return get_entry()->get_element(); }

void Node::set_tensor(const Entry::tensor_t& v) { get_entry()->set_tensor(v); }

Entry::tensor_t Node::get_tensor() const { return get_entry()->get_tensor(); }

void Node::set_block(const Entry::block_t& v) { get_entry()->set_block(v); }

Entry::block_t Node::get_block() const { return get_entry()->get_block(); }

// as Tree
// as container

Node Node::parent() const
{
    auto pos = m_prefix_.rfind("/");
    if (pos == std::string::npos)
    {
        return Node(m_entry_->parent());
    }
    else
    {
        return Node(m_entry_->parent(), m_prefix_.substr(0, pos));
    }
}

Node const& Node::self() const { return *this; }

Node& Node::self() { return *this; }

Node::range Node::children() const { return range{m_entry_->first_child(), nullptr}; }

// as array

Node Node::push_back() { return Node{get_entry()->push_back()}; }

Node Node::pop_back() { return Node{get_entry()->pop_back()}; }

Node Node::operator[](int idx) { return Node{get_entry()->item(idx)}; }

Node Node::operator[](int idx) const { return Node{get_entry()->item(idx)}; }

// as map
// @note : map is unordered
bool Node::has_a(const std::string& name) const { return m_entry_->find(join_path(m_prefix_, name)) != nullptr; }

Node Node::find(const std::string& name) const { return Node(m_entry_->find(join_path(m_prefix_, name))); }

Node Node::operator[](const std::string& name) { return Node(m_entry_, join_path(m_prefix_, name)); }

Node Node::insert(const std::string& name) { return Node(m_entry_->insert(join_path(m_prefix_, name))); }

void Node::remove(const std::string& name) { m_entry_->remove(join_path(m_prefix_, name)); }

//-------------------------------------------------------------------
// level 2
size_t Node::depth() const { return m_entry_ == nullptr ? 0 : parent().depth() + 1; }

size_t Node::height() const
{
    NOT_IMPLEMENTED;
    return 0;
}

Node::range Node::slibings() const { return Node::range{}; }

Node::range Node::ancestor() const
{
    NOT_IMPLEMENTED;
    return Node::range{};
}

Node::range Node::descendants() const
{
    NOT_IMPLEMENTED;
    return Node::range{};
}

Node::range Node::leaves() const
{
    NOT_IMPLEMENTED;
    return Node::range{};
}

Node::range Node::shortest_path(Node const& target) const
{
    NOT_IMPLEMENTED;
    return Node::range{};
}

ptrdiff_t Node::distance(const this_type& target) const
{
    NOT_IMPLEMENTED;
    return 0;
}

Node load(const std::string& uri) { NOT_IMPLEMENTED; }

void save(const Node&, const std::string& uri) { NOT_IMPLEMENTED; }

Node load(const std::istream&, const std::string& format) { NOT_IMPLEMENTED; }

void save(const Node&, const std::ostream&, const std::string& format) { NOT_IMPLEMENTED; }

std::string to_string(Entry::element_t const& s)
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

Entry::element_t from_string(const std::string& s)
{
    NOT_IMPLEMENTED;
}

std::ostream& fancy_print(std::ostream& os, const Node& entry, int indent = 0)
{
    if (entry.type() == Entry::Type::Element)
    {
        auto v = entry.get_element();

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

std::ostream& operator<<(std::ostream& os, Node const& entry) { return fancy_print(os, entry, 0); }

Node::iterator::iterator() : m_iterator_(nullptr) {}

Node::iterator::iterator(const std::shared_ptr<Entry::iterator>& it) : m_iterator_(it) {}

Node::iterator::iterator(const iterator& other) : m_iterator_(other.m_iterator_->copy()) {}

Node::iterator::iterator(iterator&& other) : m_iterator_(other.m_iterator_) { other.m_iterator_.reset(); }

bool Node::iterator::operator==(iterator const& other) const { return m_iterator_ == other.m_iterator_ || (m_iterator_ != nullptr && m_iterator_->equal(*other.m_iterator_)); }

bool Node::iterator::operator!=(iterator const& other) const { return !(operator==(other)); }

Node Node::iterator::operator*() { return Node(m_iterator_ == nullptr ? nullptr : m_iterator_->get()); }

std::unique_ptr<Node> Node::iterator::operator->() { return std::make_unique<Node>(m_iterator_ == nullptr ? nullptr : m_iterator_->get()); }

Node::iterator& Node::iterator::operator++()
{
    if (m_iterator_ != nullptr)
    {
        m_iterator_->next();
    }
    return *this;
}

Node::iterator Node::iterator::operator++(int)
{
    Node::iterator res(*this);
    if (m_iterator_ != nullptr)
    {
        m_iterator_->next();
    }
    return std::move(res);
}

} // namespace sp