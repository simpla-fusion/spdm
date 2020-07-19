#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include <any>
#include <array>
#include <complex>
#include <experimental/propagate_const>
#include <functional>
#include <map>
#include <memory>
#include <variant>
namespace sp
{
struct type_desc;

struct XPath;

class EntryInterface;

class Entry
{
private:
    // std::experimental::propagate_const<>
    std::unique_ptr<EntryInterface> m_pimpl_;

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
    template <typename U>
    class Cursor
    {

    public:
        typedef Cursor<U> this_type;
        typedef U value_type;
        typedef value_type* pointer;
        typedef value_type& reference;

        typedef std::function<pointer(pointer)> traversal_fun;

        Cursor(pointer p = nullptr) : m_current_(p), f_next_([](pointer) -> pointer { return nullptr; }) {}
        Cursor(pointer p, traversal_fun const& next) : m_current_(p), f_next_(next) {}
        Cursor(const this_type& other) : m_current_(other.m_current_), f_next_(other.f_next_) {}
        Cursor(this_type&& other) : m_current_(other.m_current_), f_next_(std::move(other.f_next_)) { other.m_current_ = nullptr; }
        ~Cursor() = default;

        this_type& operator=(this_type const& other)
        {
            this_type(other).swap(*this);
            return *this;
        }
        void swap(this_type& other)
        {
            std::swap(m_current_, other.m_current_);
            std::swap(f_next_, other.f_next_);
        }

        reference operator*() const { return *m_current_; }

        pointer operator->() const { return m_current_; }

        operator bool() const { return m_current_ != nullptr; }

        bool operator==(const this_type& other) const { return m_current_ == other.m_current_; }
        bool operator!=(const this_type& other) const { return m_current_ != other.m_current_; }
        bool operator==(pointer p) const { return m_current_ == p; }
        bool operator!=(pointer p) const { return m_current_ != p; }

        this_type& operator++()
        {
            m_current_ = f_next_(m_current_);
            return *this;
        }

        this_type operator++(int)
        {
            this_type res(*this);
            m_current_ = f_next_(m_current_);
            return res;
        }

    private:
        pointer m_current_;
        traversal_fun f_next_;
    };

    typedef Cursor<Entry> cursor;
    typedef Cursor<const Entry> const_cursor;

    friend class EntryInterface;

    typedef std::pair<cursor, cursor> range;

    typedef Entry this_type;

    typedef std::variant<std::string,
                         bool, int, double,
                         std::complex<double>,
                         std::array<int, 3>,
                         std::array<double, 3>>
        single_t;

    typedef std::tuple<std::shared_ptr<void> /* data ponter*/,
                       const std::type_info& /* type information */,
                       std::vector<size_t> /* dimensions */>
        tensor_t;

    typedef std::tuple<std::shared_ptr<void> /* data ponter*/,
                       type_desc /* type description*/,
                       std::vector<size_t> /* shapes */,
                       std::vector<size_t> /* offset */,
                       std::vector<size_t> /* strides */,
                       std::vector<size_t> /* dimensions */
                       >
        block_t;

    Entry();

    Entry(const this_type&);

    Entry(this_type&&);

    ~Entry();

    void swap(this_type&);

    this_type& operator=(this_type const& other);

    bool operator==(this_type const& other) const;

    // metadata
    Type type() const;
    bool is_null() const;
    bool is_single() const;
    bool is_tensor() const;
    bool is_block() const;
    bool is_array() const;
    bool is_object() const;

    //

    std::string prefix() const;

    // attributes

    bool has_attribute(const std::string& name) const;

    const single_t get_attribute(const std::string& name);

    void set_attribute(const std::string& name, const single_t& value);

    void remove_attribute(const std::string& name);

    std::map<std::string, single_t> attributes() const;

    //----------------------------------------------------------------------------------
    // level 0
    //
    // as leaf

    void set_single(const single_t&);
    single_t get_single() const;

    template <typename V>
    void set_value(const V& v) { set_single(single_t(v)); };
    template <typename V>
    V get_value() const { return std::get<V>(get_single()); }

    void set_tensor(const tensor_t&);
    tensor_t get_tensor() const;

    void set_block(const block_t&);
    block_t get_block() const;

    template <typename... Args>
    void set_block(Args&&... args) { return selt_block(std::make_tuple(std::forward<Args>(args)...)); };

    // as Tree
    cursor parent() const;

    const_cursor self() const;

    cursor self();

    cursor next() const;

    cursor first_child() const;

    cursor last_child() const;

    range children() const;

    // as container
    size_t size() const;

    typedef std::function<bool(this_type const&)> pred_fun;

    range find(const pred_fun& pred);

    void erase(const cursor&);

    void erase_if(const pred_fun& pred);

    void erase_if(const range&, const pred_fun& pred);

    void clear();

    // as vector

    cursor at(int); // access specified child with bounds checking

    Entry& operator[](int); // access  specified child

    cursor push_back();

    cursor push_back(const Entry&);

    cursor push_back(Entry&&);

    template <typename... Args>
    cursor emplace_back(Args&&... args) { return push_back(std::move(this_type(std::forward<Args>(args)...))); };

    Entry pop_back();

    // as map
    // @note : map is unordered

    bool has_a(const std::string& key);

    cursor find(const std::string& key);

    cursor at(const std::string& key); // access specified child with bounds checking

    Entry& operator[](const std::string&); // access or insert specified child

    cursor insert(const std::string& key); // if key is not exists then insert node at key else return entry at key

    cursor insert(const std::string& pos, const Entry& other);

    cursor insert(const std::string& pos, Entry&& other);

    template <typename... Args>
    cursor emplace(const std::string& key, Args&&... args)
    {
        cursor p = find(key);
        if (!p)
        {
            p = insert(key, std::move(Entry(std::forward<Args>(args)...)));
        }
        return p;
    }

    Entry erase(const std::string&);

    //-------------------------------------------------------------------
    // level 1
    // xpath

    Entry fetch(const XPath&) const;

    bool update(const XPath&, Entry&&);

    bool update(const XPath&, const Entry&);

    //-------------------------------------------------------------------
    // level 2

    size_t depth() const; // parent.depth +1

    size_t height() const; // max(children.height) +1

    range slibings() const; // return slibings

    range ancestor() const; // return ancestor

    range descendants() const; // return descendants

    range leaves() const; // return leave nodes in traversal order

    range shortest_path(cursor const& target) const; // return the shortest path to target

    ptrdiff_t distance(const this_type& target) const; // lenght of shortest path to target
};

} // namespace sp

#endif // SP_ENTRY_H_
