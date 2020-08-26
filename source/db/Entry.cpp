#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"

namespace sp::utility
{

std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    os << std::setw(indent * tab) << " " << entry.get_value();
    return os;
}
} // namespace sp::utility

namespace sp::db
{

std::ostream& operator<<(std::ostream& os, Entry const& entry) { return sp::utility::fancy_print(os, entry, 0); }

//===========================================================================================================
// Entry

Entry::Entry(std::initializer_list<std::pair<std::string, Node>> init, Path p) : m_root_(nullptr), m_path_(std::move(p))
{
    auto& obj = root();
    for (auto&& item : init)
    {
        obj.update_value(item.first, Node(item.second));
    }
}

Entry::Entry(std::shared_ptr<NodeObject> root, Path p) : m_root_(root), m_path_(std::move(p)) {}

Entry::Entry(const Entry& other) : m_root_(other.m_root_), m_path_(other.m_path_) {}

Entry::Entry(Entry&& other) : m_root_(std::move(other.m_root_)), m_path_(std::move(other.m_path_)) {}

void Entry::swap(Entry& other)
{
    std::swap(m_root_, other.m_root_);
    std::swap(m_path_, other.m_path_);
}

void Entry::load(const Node& opt) { m_root_ = NodeObject::create(opt); }

void Entry::save(const Node& opt) const { root().save(opt); }

NodeObject& Entry::root()
{
    if (m_root_ == nullptr)
    {
        m_root_ = NodeObject::create({});
    }
    return *m_root_;
}

const NodeObject& Entry::root() const { return *m_root_; }

std::size_t Entry::type() const { return root().fetch(m_path_, {{"$type", true}}).as<int>(); }

void Entry::reset()
{
    m_root_ = NodeObject::create();
    m_path_.clear();
}

bool Entry::is_null() const { return m_root_ == nullptr || type() == Node::tags::Null; }

bool Entry::empty() const { return is_null() || size() == 0; }

size_t Entry::size() const { return root().fetch(m_path_, {{"$size", true}}, {}).as<size_t>(); }

bool Entry::operator==(const Entry& other) const
{
    return m_root_ == other.m_root_ || root().is_same(*other.m_root_) && m_path_ == other.m_path_;
}

// std::pair<std::shared_ptr<const NodeObject>, Path> Entry::full_path() const
// {
//     NOT_IMPLEMENTED;
//     return std::pair<std::shared_ptr<const NodeObject>, Path>{};
// }

// std::pair<std::shared_ptr<NodeObject>, Path> Entry::full_path()
// {
//     NOT_IMPLEMENTED;
//     return std::pair<std::shared_ptr<NodeObject>, Path>{};
// }

//-----------------------------------------------------------------------------------------------------------
using namespace std::string_literals;

void Entry::set_value(const Node& v) { root().update(m_path_, std::move(v)); }

Node Entry::get_value() const { return root().fetch(m_path_, {}); }

NodeObject& Entry::as_object() { return m_path_.length() == 0 ? *m_root_ : *root().merge(m_path_, Node{NodeObject::create()}).as<Node::tags::Object>(); }

const NodeObject& Entry::as_object() const { return m_path_.length() == 0 ? *m_root_ : *root().fetch(m_path_).as<Node::tags::Object>(); }

NodeArray& Entry::as_array() { return *root().merge(m_path_, Node(NodeArray::create())).as<Node::tags::Array>(); }

const NodeArray& Entry::as_array() const { return *root().fetch(m_path_).as<Node::tags::Array>(); }

//-----------------------------------------------------------------------------------------------------------

void Entry::resize(std::size_t num) { as_array().resize(num); }

Node Entry::pop_back() { return as_array().pop_back(); }

Entry Entry::push_back(Node v)
{
    auto& a = as_array();
    a.push_back(std::move(v));
    return Entry{m_root_, Path(m_path_).join(a.size() - 1)};
}

Cursor<Node> Entry::Entry::children()
{
    NOT_IMPLEMENTED;
    Cursor<Node> res;
    // std::visit(
    //     sp::traits::overloaded{
    //         [&](std::variant_alternative_t<Node::tags::Object, value_type>& object_p) { object_p->children().swap(res); },
    //         [&](std::variant_alternative_t<Node::tags::Array, value_type>& array_p) { array_p->children().swap(res); },
    //         [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
    //     fetch());
    return std::move(res);
}

Cursor<const Node> Entry::Entry::children() const
{
    NOT_IMPLEMENTED;
    Cursor<const Node> res;
    // std::visit(
    //     sp::traits::overloaded{
    //         [&](const std::variant_alternative_t<Node::tags::Object, value_type>& object_p) { std::const_pointer_cast<const NodeObject>(object_p)->children().swap(res); },
    //         [&](const std::variant_alternative_t<Node::tags::Array, value_type>& array_p) { std::const_pointer_cast<const NodeArray>(array_p)->children().swap(res); },
    //         [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
    //     fetch());
    return std::move(res);
}

// void Entry::for_each(std::function<void(const Path::Segment&, Node&)> const&) { NOT_IMPLEMENTED; }

} // namespace sp::db
