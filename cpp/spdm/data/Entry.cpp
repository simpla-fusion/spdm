#include "Entry.h"
#include "../utility/Factory.h"
#include "../utility/TypeTraits.h"
#include "../utility/fancy_print.h"
#include "DataBlock.h"
#include <cassert>
namespace sp::utility
{

std::ostream& fancy_print(std::ostream& os, const sp::db::Entry& entry, int indent = 0, int tab = 4)
{
    os << std::setw(indent * tab) << " " << entry.fetch();
    return os;
}
} // namespace sp::utility

namespace sp::db
{

std::ostream& operator<<(std::ostream& os, Entry const& entry) { return sp::utility::fancy_print(os, entry, 0); }

//===========================================================================================================
// Entry
Entry::Entry(const Node& opt) { load(opt); }

Entry::Entry(std::initializer_list<Node> init, Path p) : Entry(Node(init)) { m_path_.swap(p); }

Entry::Entry(const std::shared_ptr<NodeObject>& r, Path p) : m_root_(r), m_path_(std::move(p)) {}

Entry::Entry(const Entry& other) : m_root_(other.m_root_), m_path_(other.m_path_) {}

Entry::Entry(Entry&& other) : m_root_(std::move(other.m_root_)), m_path_(std::move(other.m_path_)) {}

void Entry::swap(Entry& other)
{
    m_root_.swap(other.m_root_);
    m_path_.swap(other.m_path_);
}

Entry& Entry::operator=(const Entry& other)
{
    Entry(other).swap(*this);
    return *this;
}

void Entry::load(const Node& opt)
{

    if (m_root_ == nullptr)
    {
        m_root_ = NodeObject::create(opt);
    }
    else
    {
        m_root_->load(opt);
    }
}

void Entry::save(const Node& opt) const { root()->save(opt); }

std::shared_ptr<NodeObject> Entry::root()
{
    if (m_root_ == nullptr)
    {
        m_root_ = NodeObject::create();
    }
    return m_root_;
}

std::shared_ptr<const NodeObject> Entry::root() const
{
    assert(m_root_ != nullptr);
    return m_root_;
}

void Entry::reset()
{
    m_root_ = nullptr;
    m_path_.clear();
}
//-------------------------------

Node Entry::update(int op, Node const& args) { return root()->update(m_path_, op, args); }

const Node Entry::fetch(int op, Node const& args) const { return root()->fetch(m_path_, op, args); }

//-------------------------------

size_t Entry::type() const { return fetch(Node::ops::TYPE).get_value<size_t>(); }

bool Entry::is_null() const { return m_root_ == nullptr || type() == Node::tags::Null; }

bool Entry::empty() const { return is_null() || count() == 0; }

size_t Entry::count() const { return fetch(Node::ops::COUNT).get_value<size_t>(size_t(0)); }

bool Entry::same_as(const Entry& other) const
{
    return m_root_ == other.m_root_ || root()->is_same(*other.m_root_) && m_path_ == other.m_path_;
}

//-----------------------------------------------------------------------------------------------------------
using namespace std::string_literals;

void Entry::resize(int num) { update(Node::ops::RESIZE, num); }

Node Entry::pop_back() { return update(Node::ops::POP_BACK); }

Entry Entry::push_back(Node v)
{
    int idx = update(Node::ops::PUSH_BACK, v).get_value<int>();
    return Entry(m_root_, Path(m_path_).join(idx));
}

Cursor<Node> Entry::Entry::children()
{
    NOT_IMPLEMENTED;
    return Cursor<Node>{};
}

Cursor<const Node> Entry::Entry::children() const
{
    NOT_IMPLEMENTED;
    return Cursor<const Node>{};
}

// void Entry::for_each(std::function<void(const Path::Segment&, Node&)> const&) { NOT_IMPLEMENTED; }

} // namespace sp::db
