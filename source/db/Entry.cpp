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

//-----------------------------------------------------------------------------------------------------------

Entry::Entry(std::shared_ptr<NodeObject> r, Path p) : m_root_(std::move(r)), m_path_(std::move(p)) {}

Entry::Entry(const Entry& other) : m_root_(other.m_root_), m_path_(other.m_path_) {}

Entry::Entry(Entry&& other) : m_root_(std::move(other.m_root_)), m_path_(std::move(other.m_path_)) {}

void Entry::swap(Entry& other)
{
    std::swap(m_root_, other.m_root_);
    std::swap(m_path_, other.m_path_);
}

Entry Entry::create(const std::string& url)
{
    return Entry{NodeObject::create(url), Path{}};
}

void Entry::load(const std::string& url)
{
    if (m_root_ == nullptr)
    {
        m_root_ = NodeObject::create(url);
    }
    else
    {
        m_root_->load(url);
    }
}

void Entry::save(const std::string& url) const
{
    if (m_root_ != nullptr)
    {
        m_root_->save(Path(m_path_).join(url).str());
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

std::size_t Entry::type() const { return root().find(m_path_).index(); }

void Entry::reset()
{
    m_root_.reset();
    m_path_.clear();
}

bool Entry::is_null() const { return m_root_ == nullptr || type() == value_type_tags::Null; }

bool Entry::empty() const { return is_null() || size() == 0; }

size_t Entry::size() const
{
    size_t res = 0;
    auto tmp = root().find(m_path_);

    switch (tmp.index())
    {
    case tree_node_tags::Object:
        res = std::get<tree_node_tags::Object>(tmp)->size();
        break;
    case tree_node_tags::Array:
        res = std::get<tree_node_tags::Array>(tmp)->size();
        break;
    default:
        res = 0;
    }
    return res;
}

bool Entry::operator==(const Entry& other) const
{
    NOT_IMPLEMENTED;
    return false;
}

NodeObject& Entry::root()
{
    if (m_root_ == nullptr)
    {
        m_root_ = NodeObject::create();
    }

    return *m_root_;
}

const NodeObject& Entry::root() const
{
    if (m_root_ == nullptr)
    {
        RUNTIME_ERROR << "Root is not defined!";
    }
    return *m_root_;
}

std::pair<std::shared_ptr<const NodeObject>, Path> Entry::full_path() const
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<const NodeObject>, Path>{};
}

std::pair<std::shared_ptr<NodeObject>, Path> Entry::full_path()
{
    NOT_IMPLEMENTED;
    return std::pair<std::shared_ptr<NodeObject>, Path>{};
}

//-----------------------------------------------------------------------------------------------------------

void Entry::set_value(tree_node_type v) { root().update(m_path_, std::move(v)); }

tree_node_type Entry::get_value() const { return root().find(m_path_); }

NodeObject& Entry::as_object() { return *std::get<tree_node_tags::Object>(root().insert(m_path_, tree_node_type{NodeObject::create()})); }

const NodeObject& Entry::as_object() const { return *std::const_pointer_cast<const NodeObject>(std::get<tree_node_tags::Object>(root().find(m_path_))); }

NodeArray& Entry::as_array() { return *std::get<tree_node_tags::Array>(root().insert(m_path_, tree_node_type{NodeArray::create()})); }

const NodeArray& Entry::as_array() const { return *std::const_pointer_cast<const NodeArray>(std::get<tree_node_tags::Array>(root().find(m_path_))); }

//-----------------------------------------------------------------------------------------------------------

void Entry::resize(std::size_t num) { as_array().resize(num); }

tree_node_type Entry::pop_back() { return as_array().pop_back(); }

Entry Entry::push_back(tree_node_type v)
{
    auto a = as_array();
    a.push_back(std::move(v));
    return Entry{m_root_, Path(m_path_).join(a.size() - 1)};
}

Cursor<tree_node_type> Entry::Entry::children()
{
    NOT_IMPLEMENTED;
    Cursor<tree_node_type> res;
    // std::visit(
    //     sp::traits::overloaded{
    //         [&](std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { object_p->children().swap(res); },
    //         [&](std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { array_p->children().swap(res); },
    //         [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
    //     fetch());
    return std::move(res);
}

Cursor<const tree_node_type> Entry::Entry::children() const
{
    NOT_IMPLEMENTED;
    Cursor<const tree_node_type> res;
    // std::visit(
    //     sp::traits::overloaded{
    //         [&](const std::variant_alternative_t<value_type_tags::Object, value_type>& object_p) { std::const_pointer_cast<const NodeObject>(object_p)->children().swap(res); },
    //         [&](const std::variant_alternative_t<value_type_tags::Array, value_type>& array_p) { std::const_pointer_cast<const NodeArray>(array_p)->children().swap(res); },
    //         [&](auto&&) { RUNTIME_ERROR << "illegal type!"; }},
    //     fetch());
    return std::move(res);
}

// void Entry::for_each(std::function<void(const Path::Segment&, tree_node_type&)> const&) { NOT_IMPLEMENTED; }

void Entry::for_each(std::function<void(const Path::Segment&, tree_node_type)> const&) const { NOT_IMPLEMENTED; }

} // namespace sp::db
