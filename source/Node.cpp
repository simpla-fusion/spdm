#include "Node.h"
// #include "Entry.h"
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
template <>
class HierarchicalTreeObjectPolicy<Node>
{
public:
    typedef Node node_type;

    HierarchicalTreeObjectPolicy(const std::string&){};
    HierarchicalTreeObjectPolicy() = default;
    HierarchicalTreeObjectPolicy(HierarchicalTreeObjectPolicy&&) = default;
    HierarchicalTreeObjectPolicy(const HierarchicalTreeObjectPolicy&) = default;
    ~HierarchicalTreeObjectPolicy() = default;

    void swap(HierarchicalTreeObjectPolicy& other) { m_data_.swap(other.m_data_); }

    HierarchicalTreeObjectPolicy& operator=(HierarchicalTreeObjectPolicy const& other)
    {
        HierarchicalTreeObjectPolicy(other).swap(*this);
        return *this;
    }
   
    size_t size() const { return m_data_.size(); }

    void clear() { m_data_.clear(); }

    const node_type& at(const std::string& key) const { return m_data_.at(key); };

    template <typename... Args>
    node_type& try_emplace(const std::string& key, Args&&... args) { return m_data_.try_emplace(key, std::forward<Args>(args)...).first->second; }

    node_type& insert(const std::string& path) { return try_emplace(path); }

    void erase(const std::string& key) { m_data_.erase(key); }

    // class iterator;

    template <typename... Args>
    decltype(auto) find(const std::string& key, Args&&... args) const { return m_data_.find(key); }

    template <typename... Args>
    decltype(auto) find(const std::string& key, Args&&... args) { return m_data_.find(key); }

    decltype(auto) begin() { return m_data_.begin(); }

    decltype(auto) end() { return m_data_.end(); }

    decltype(auto) begin() const { return m_data_.cbegin(); }

    decltype(auto) end() const { return m_data_.cend(); }

    // template <typename... Args>
    // const iterator find(const std::string&, Args&&... args) const;

    // template <typename... Args>
    // int erase(Args&&... args);

private:
    std::map<std::string, node_type> m_data_;
};

template <>
class HierarchicalTreeArrayPolicy<Node>
{
public:
    typedef Node node_type;

    HierarchicalTreeArrayPolicy() = default;
    HierarchicalTreeArrayPolicy(HierarchicalTreeArrayPolicy&&) = default;
    HierarchicalTreeArrayPolicy(const HierarchicalTreeArrayPolicy&) = default;
    ~HierarchicalTreeArrayPolicy() = default;

    void swap(HierarchicalTreeArrayPolicy& other) { m_data_.swap(other.m_data_); }

    HierarchicalTreeArrayPolicy& operator=(HierarchicalTreeArrayPolicy const& other)
    {
        HierarchicalTreeArrayPolicy(other).swap(*this);
        return *this;
    }

    size_t size() const { return m_data_.size(); }

    void resize(size_t s) { m_data_.resize(s); }

    void clear() { m_data_.clear(); }

    template <typename... Args>
    node_type& emplace_back(Args&&... args)
    {
        m_data_.emplace_back(std::forward<Args>(args)...);
        return m_data_.back();
    }

    void pop_back() { m_data_.pop_back(); }

    node_type& at(int idx) { return m_data_.at(idx); }

    const node_type& at(int idx) const { return m_data_.at(idx); }

    decltype(auto) begin() { return m_data_.begin(); }

    decltype(auto) end() { return m_data_.end(); }

    decltype(auto) begin() const { return m_data_.cbegin(); }

    decltype(auto) end() const { return m_data_.cend(); }

private:
    std::vector<node_type> m_data_;
};


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

std::ostream& fancy_print(std::ostream& os, const Node& entry, int indent = 0, int tab = 4)
{

    if (entry.type() == Entry::NodeType::Element)
    {
        os << to_string(entry.get_element());
    }
    else if (entry.type() == Entry::NodeType::Array)
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
    else if (entry.type() == Entry::NodeType::Object)
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