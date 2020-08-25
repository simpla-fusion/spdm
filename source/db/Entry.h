#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
#include "../utility/TypeTraits.h"
#include "Cursor.h"
#include "DataBlock.h"
#include "Node.h"
#include "XPath.h"
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace sp::db
{

class Entry
{
    std::shared_ptr<NodeObject> m_root_;
    Path m_path_;

public:
    typedef tree_node_type value_type;

    typedef tree_node_tags value_type_tags;

    Entry(std::shared_ptr<NodeObject> r = nullptr, Path p = {});

    Entry(const Entry& other);

    Entry(Entry&& other);

    ~Entry() = default;

    void swap(Entry& other);

    Entry& operator=(const Entry& other)
    {
        Entry(other).swap(*this);
        return *this;
    }

    static Entry create(const std::string&);

    void load(const std::string&);

    void save(const std::string&) const;

    //-------------------------------------------------------------------------

    std::size_t type() const;

    void reset();

    bool is_null() const;

    bool empty() const;

    size_t size() const;

    NodeObject& root();

    const NodeObject& root() const;

    const Path& path() const { return m_path_; }

    std::pair<std::shared_ptr<const NodeObject>, Path> full_path() const;

    std::pair<std::shared_ptr<NodeObject>, Path> full_path();

    //-------------------------------------------------------------------------

    NodeObject& as_object();
    const NodeObject& as_object() const;

    NodeArray& as_array();
    const NodeArray& as_array() const;

    void set_value(value_type v);
    tree_node_type get_value() const;

    template <typename V, typename First, typename... Others>
    void as(First&& first, Others&&... others) { set_value(value_type(std::in_place_type_t<V>(), std::forward<First>(first), std::forward<Others>(others)...)); }

    template <typename V>
    V as() const { return traits::convert<V>(get_value()); }

    template <typename V>
    Entry& operator=(const V& v)
    {
        as<V>(v);
        return *this;
    }

    Entry& operator=(const char* v)
    {
        as<std::string>(v);
        return *this;
    }

    //-------------------------------------------------------------------------
    // access
    template <typename... Args>
    Entry at(Args&&... args) & { return Entry{root().shared_from_this(), m_path_.join(std::forward<Args>(args)...)}; }

    template <typename... Args>
    Entry at(Args&&... args) && { return Entry{root().shared_from_this(), std::move(m_path_).join(std::forward<Args>(args)...)}; }

    template <typename... Args>
    Entry at(Args&&... args) const& { return Entry{m_root_, m_path_.join(std::forward<Args>(args)...)}; }

    template <typename T>
    inline Entry operator[](const T& idx) & { return at(idx); }
    template <typename T>
    inline Entry operator[](const T& idx) && { return at(idx); }
    template <typename T>
    inline Entry operator[](const T& idx) const& { return at(idx); }

    inline Entry slice(int start, int stop, int step = 1) & { return at(std::make_tuple(start, stop, step)); }
    inline Entry slice(int start, int stop, int step = 1) && { return at(std::make_tuple(start, stop, step)); }
    inline Entry slice(int start, int stop, int step = 1) const& { return at(std::make_tuple(start, stop, step)); }

    //-------------------------------------------------------------------------

    void resize(std::size_t num);

    tree_node_type pop_back();

    Entry push_back(tree_node_type v = {});

    Cursor<tree_node_type> children();

    Cursor<const tree_node_type> children() const;

    void for_each(std::function<void(const Path::Segment&, tree_node_type)> const&) const;

    //------------------------------------------------------------------------------------

    bool operator==(const Entry& other) const;

private:
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_