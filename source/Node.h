#ifndef SP_NODE_H_
#define SP_NODE_H_
#include <any>
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <variant>
#include <vector>

namespace sp
{
struct XPath;

class Entry;

class Node
{
private:
    std::string m_prefix_;
    std::shared_ptr<Entry> m_entry_;

    std::shared_ptr<Entry> get_entry() const;
    std::shared_ptr<Entry> get_entry();

public:
    enum Type
    {
        Null = 0,
        Single = 1,
        Tensor = 2,
        Block = 3,
        Array = 4,
        Object = 5
    };

    typedef std::variant<std::string,
                         bool, int, double,
                         std::complex<double>,
                         std::array<int, 3>,
                         std::array<double, 3>>
        element_t;

    typedef std::tuple<std::shared_ptr<void> /* data ponter*/,
                       const std::type_info& /* type information */,
                       std::vector<size_t> /* dimensions */>
        tensor_t;

    typedef std::tuple<std::shared_ptr<void> /* data ponter*/,
                       std::string /* type description*/,
                       std::vector<size_t> /* shapes */,
                       std::vector<size_t> /* offset */,
                       std::vector<size_t> /* strides */,
                       std::vector<size_t> /* dimensions */
                       >
        block_t;

    friend class Entry;

    typedef Node this_type;

    class iterator;
    class range;

    Node();

    explicit Node(const std::string& uri);

    explicit Node(const std::shared_ptr<Entry>& p,
                  const std::string& prefix = "");

    explicit Node(Entry* p,
                  const std::string& prefix = "");

    Node(const this_type&);

    Node(this_type&&);

    ~Node();

    void swap(this_type&);

    this_type& operator=(this_type const& other);

    bool operator==(this_type const& other) const;

    operator bool() const { return !is_null(); }

    void resolve();

    // metadata
    Type type() const;
    bool is_null() const;
    bool is_single() const;
    bool is_tensor() const;
    bool is_block() const;
    bool is_array() const;
    bool is_object() const;

    bool is_root() const;
    bool is_leaf() const;

    //

    std::string full_path() const;

    std::string relative_path() const;

    // attributes

    bool has_attribute(const std::string& name) const;

    const element_t get_attribute_raw(const std::string& name) const;

    void set_attribute_raw(const std::string& name, const element_t& value);

    void remove_attribute(const std::string& name);

    template <typename V>
    const element_t get_attribute(const std::string& name)
    {
        return std::get<V>(get_attribute_raw(name));
    };

    void set_attribute(const std::string& name, const char* value)
    {
        set_attribute_raw(name, element_t{std::string(value)});
    }

    template <typename V>
    void set_attribute(const std::string& name, const V& value)
    {
        set_attribute_raw(name, element_t{value});
    }

    std::map<std::string, element_t> attributes() const;

    //----------------------------------------------------------------------------------
    // level 0
    //
    // as leaf

    void set_single(const element_t&);

    element_t get_single() const;

    template <typename V>
    void set_value(const V& v) { set_single(element_t(v)); };

    template <typename V>
    V get_value() const { return std::get<V>(get_single()); }

    void set_tensor(const tensor_t&);

    tensor_t get_tensor() const;

    void set_block(const block_t&);

    block_t get_block() const;

    template <typename... Args>
    void set_block(Args&&... args) { return selt_block(std::make_tuple(std::forward<Args>(args)...)); };

    // as Tree
    // as container

    const Node& self() const;

    Node& self();

    Node parent() const;

    range children() const;

    range children();

    void clear();

    // as array

    Node operator[](int); // access  specified child

    Node operator[](int) const; // access  specified child

    Node push_back(); // append new item

    Node pop_back(); // remove and return last item

    // as object
    // @note : map is unordered

    Node insert(const std::string& key); // if key is not exists then insert node at key else return Node at key

    bool has_a(const std::string& key) const;

    Node find(const std::string& key) const;

    Node operator[](const char* c) const { return operator[](std::string(c)); }

    Node operator[](const char* c) { return operator[](std::string(c)); }

    Node operator[](const std::string&) const; // access  specified child

    Node operator[](const std::string&); // access or insert specified child

    void remove(const std::string&);

    //-------------------------------------------------------------------
    // level 1
    // xpath

    Node insert(const XPath&);

    range find(const XPath&) const;

    typedef std::function<bool(const Node&)> pred_fun;

    range find(const pred_fun&) const;

    int update(range&, const Node&);

    int remove(range&);

    //-------------------------------------------------------------------
    // level 2

    size_t depth() const; // parent.depth +1

    size_t height() const; // max(children.height) +1

    range slibings() const; // return slibings

    range ancestor() const; // return ancestor

    range descendants() const; // return descendants

    range leaves() const; // return leave nodes in traversal order

    range shortest_path(Node const& target) const; // return the shortest path to target

    ptrdiff_t distance(const this_type& target) const; // lenght of shortest path to target
};

class Node::iterator
{
public:
    iterator();

    ~iterator() = default;

    iterator(const std::shared_ptr<Entry>&);

    iterator(const iterator&);

    iterator(iterator&&);

    bool operator==(iterator const& other) const;

    bool operator!=(iterator const& other) const;

    Node operator*();

    std::unique_ptr<Node> operator->();

    iterator& operator++();

    iterator operator++(int);

private:
    std::shared_ptr<Entry> m_entry_;
};

class Node::range : public std::pair<Node::iterator, Node::iterator>
{
public:
    typedef std::pair<Node::iterator, Node::iterator> base_type;

    using base_type::pair;

    using base_type::first;

    using base_type::second;

    range() = default;

    ~range() = default;

    Node::iterator begin() { return first; }

    Node::iterator end() { return second; }
};

std::ostream& operator<<(std::ostream& os, Node const& Node);

} // namespace sp

#endif // SP_NODE_H_
