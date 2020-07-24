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

Node::Node(const std::shared_ptr<Entry>& p, const std::string& path) : m_entry_(p), m_path_(path) {}

Node::Node(const std::string& uri) : Node(Entry::create(uri), "") {}

Node::Node(const Node& other) : m_path_(other.m_path_), m_entry_(other.m_entry_) {}

Node::Node(Node&& other) : m_path_(other.m_path_), m_entry_(other.m_entry_)
{
    other.m_entry_.reset();
    other.m_path_ = "";
}

Node::~Node() {}

void Node::swap(this_type& other)
{
    std::swap(m_path_, other.m_path_);
    std::swap(m_entry_, other.m_entry_);
}

Node& Node::operator=(this_type const& other)
{
    this_type(other).swap(*this);
    return *this;
}

//

std::shared_ptr<Entry> Node::self() const
{
    if (m_entry_ == nullptr)
    {
        throw std::out_of_range("try to access empty node.");
    }
    std::shared_ptr<Entry> p = m_entry_;

    if (m_path_ != "")
    {
        p = m_entry_->find_r(m_path_);
    }

    if (p == nullptr)
    {
        throw std::out_of_range("Can not find node:" + m_path_);
    }

    return p;
}

std::shared_ptr<Entry> Node::self()
{
    if (m_entry_ == nullptr)
    {
        m_entry_ = Entry::create(m_path_);
        m_path_ = "";
    }
    else if (m_path_ != "")
    {
        m_entry_ = m_entry_->insert_r(m_path_);
        m_path_ = "";
    }

    if (m_entry_ == nullptr)
    {
        throw std::runtime_error("null entry!");
    }

    return m_entry_;
}

std::string Node::path() const { return self()->path(); }

std::string Node::name() const { return self()->name(); }
const Node& Node::value() const { return *this; }
Node& Node::value() { return *this; }

// metadata
Entry::Type Node::type() const { return m_entry_ == nullptr ? Entry::Type::Null : m_entry_->type(); }
bool Node::is_null() const { return type() == Entry::Type::Null; }
bool Node::is_element() const { return type() == Entry::Type::Element; }
bool Node::is_tensor() const { return type() == Entry::Type::Tensor; }
bool Node::is_block() const { return type() == Entry::Type::Block; }
bool Node::is_array() const { return type() == Entry::Type::Array; }
bool Node::is_object() const { return type() == Entry::Type::Object; }

bool Node::is_root() const { return m_entry_ == nullptr || m_entry_->parent() == nullptr; }
bool Node::is_leaf() const { return type() < Entry::Type::Array; };

// attributes
bool Node::has_attribute(const std::string& name) const { return self()->has_attribute(name); }

const Entry::element_t Node::get_attribute_raw(const std::string& name) const { return self()->get_attribute_raw(name); }

void Node::set_attribute_raw(const std::string& name, const Entry::element_t& value) { self()->set_attribute_raw(name, value); }

void Node::remove_attribute(const std::string& name) { self()->remove_attribute(name); }

std::map<std::string, Entry::element_t> Node::attributes() const { return self()->attributes(); }

// as leaf
void Node::set_element(const Entry::element_t& v) { self()->set_element(v); }

Entry::element_t Node::get_element() const { return self()->get_element(); }

void Node::set_tensor(const Entry::tensor_t& v) { self()->set_tensor(v); }

Entry::tensor_t Node::get_tensor() const { return self()->get_tensor(); }

void Node::set_block(const Entry::block_t& v) { self()->set_block(v); }

Entry::block_t Node::get_block() const { return self()->get_block(); }

// as Tree
// as container

Node Node::parent() const
{
    auto pos = m_path_.rfind("/");
    if (pos == std::string::npos)
    {
        return Node(m_entry_->parent());
    }
    else
    {
        return Node(m_entry_->parent(), m_path_.substr(0, pos));
    }
}

size_t Node::size() const { return self()->size(); }

Node::cursor Node::first_child() const { return cursor{self()->first_child()}; }

Node::cursor Node::next() const { return cursor{self()->next()}; }
// as array

Node Node::push_back() { return Node{self()->push_back()}; }

Node Node::pop_back() { return Node{self()->pop_back()}; }

Node Node::operator[](int idx) { return idx < 0 ? Node{push_back()} : Node{self()->item(idx)}; }

Node Node::operator[](int idx) const { return Node{self()->item(idx)}; }

// as map
// @note : map is unordered
bool Node::has_a(const std::string& name) const { return m_entry_->find(join_path(m_path_, name)) != nullptr; }

Node Node::find(const std::string& name) const { return Node(m_entry_->find(join_path(m_path_, name))); }

Node Node::operator[](const std::string& name) { return Node(m_entry_, join_path(m_path_, name)); }

Node Node::insert(const std::string& name) { return Node(m_entry_->insert(join_path(m_path_, name))); }

void Node::remove(const std::string& name) { m_entry_->remove(join_path(m_path_, name)); }

//-------------------------------------------------------------------
// level 2
size_t Node::depth() const { return m_entry_ == nullptr ? 0 : parent().depth() + 1; }

size_t Node::height() const
{
    NOT_IMPLEMENTED;
    return 0;
}

Node::cursor Node::slibings() const { return Node::cursor{}; }

Node::cursor Node::ancestor() const
{
    NOT_IMPLEMENTED;
    return Node::cursor{};
}

Node::cursor Node::descendants() const
{
    NOT_IMPLEMENTED;
    return Node::cursor{};
}

Node::cursor Node::leaves() const
{
    NOT_IMPLEMENTED;
    return Node::cursor{};
}

Node::cursor Node::shortest_path(Node const& target) const
{
    NOT_IMPLEMENTED;
    return Node::cursor{};
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

std::ostream& fancy_print(std::ostream& os, const Node& entry, int indent = 0, int tab = 4)
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
        os << "[";
        for (auto it = entry.first_child(); !it.is_null(); ++it)
        {
            os << std::endl
               << std::setw(indent * tab) << " ";
            fancy_print(os, it->value(), indent + 1, tab);
            os << ",";
        }
        os << std::endl
           << std::setw(indent * tab)
           << "]";
    }
    else if (entry.type() == Entry::Type::Object)
    {
        os << "{";
        for (auto it = entry.first_child(); !it.is_null(); ++it)
        {
            os << std::endl
               << std::setw(indent * tab) << " "
               << "\"" << it->name() << "\" : ";
            fancy_print(os, it->value(), indent + 1, tab);
            os << ",";
        }
        os << std::endl
           << std::setw(indent * tab)
           << "}";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, Node const& entry) { return fancy_print(os, entry, 0); }

Node::cursor::cursor() : m_entry_(nullptr) {}

Node::cursor::cursor(Entry* p) : m_entry_(p == nullptr ? nullptr : p->shared_from_this()) {}

Node::cursor::cursor(const std::shared_ptr<Entry>& p) : m_entry_(p) {}

Node::cursor::cursor(const cursor& other) : m_entry_(other.m_entry_->copy()) {}

Node::cursor::cursor(cursor&& other) : m_entry_(other.m_entry_) { other.m_entry_.reset(); }

bool Node::cursor::operator==(cursor const& other) const { return m_entry_ == other.m_entry_ || (m_entry_ != nullptr && m_entry_->same_as(other.m_entry_.get())); }

bool Node::cursor::operator!=(cursor const& other) const { return !(operator==(other)); }

Node Node::cursor::operator*() { return Node(m_entry_); }

bool Node::cursor::is_null() const { return m_entry_ == nullptr || m_entry_->type() == Entry::Null; }

std::unique_ptr<Node> Node::cursor::operator->() { return std::make_unique<Node>(m_entry_); }

Node::cursor& Node::cursor::operator++()
{
    m_entry_ = m_entry_->next();
    return *this;
}

Node::cursor Node::cursor::operator++(int)
{
    Node::cursor res(*this);
    m_entry_ = m_entry_->next();
    return std::move(res);
}

} // namespace sp