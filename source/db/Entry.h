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
    Entry() = default;

    Entry(std::initializer_list<std::pair<std::string, Node>> init, Path p = {});

    Entry(std::shared_ptr<NodeObject> root, Path p = {});

    Entry(const Entry& other);

    Entry(Entry&& other);

    ~Entry() = default;

    void swap(Entry& other);

    Entry& operator=(const Entry& other)
    {
        Entry(other).swap(*this);
        return *this;
    }

    static Entry create(const Node&);

    void load(const Node&);

    void save(const Node&) const;

    //-------------------------------------------------------------------------

    std::size_t type() const;

    void reset();

    bool is_null() const;

    bool empty() const;

    size_t size() const;

    NodeObject& root();

    const NodeObject& root() const;

    const Path& path() const { return m_path_; }

    std::pair<const NodeObject, Path> full_path() const;

    std::pair<NodeObject, Path> full_path();

    //-------------------------------------------------------------------------

    NodeObject& as_object();
    const NodeObject& as_object() const;

    NodeArray& as_array();
    const NodeArray& as_array() const;

    void set_value(const Node& v);
    Node get_value() const;

    template <typename V, typename First, typename... Others>
    void as(First&& first, Others&&... others) { set_value(Node{std::in_place_type_t<V>(), std::forward<First>(first), std::forward<Others>(others)...}); }

    template <typename V>
    V as() const { return get_value().as<V>(); }

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
    Entry at(Args&&... args) & { return Entry{m_root_, Path(m_path_).join(std::forward<Args>(args)...)}; }

    template <typename... Args>
    Entry at(Args&&... args) && { return Entry{m_root_, Path(m_path_).join(std::forward<Args>(args)...)}; }

    template <typename... Args>
    Entry at(Args&&... args) const& { return Entry{m_root_, Path(m_path_).join(std::forward<Args>(args)...)}; }

    template <typename T>
    inline Entry operator[](const T& idx) & { return at(idx); }
    template <typename T>
    inline Entry operator[](const T& idx) && { return at(idx); }
    template <typename T>
    inline Entry operator[](const T& idx) const& { return at(idx); }

    inline Entry slice(int start, int stop, int step = 1) & { return at(std::make_tuple(start, stop, step)); }
    inline Entry slice(int start, int stop, int step = 1) && { return at(std::make_tuple(start, stop, step)); }
    inline Entry slice(int start, int stop, int step = 1) const& { return at(std::make_tuple(start, stop, step)); }

    template <typename V>
    auto attribute(const std::string& name) const { return at("@" + name).as<V>(); }

    template <typename V, typename... Args>
    void attribute(const std::string& name, Args&&... args) { at("@" + name).as<V>(std::forward<Args>(args)...); }

    //-------------------------------------------------------------------------

    void resize(std::size_t num);

    Node pop_back();

    Entry push_back(Node v = {});

    Cursor<Node> children();

    Cursor<const Node> children() const;

    //------------------------------------------------------------------------------------

    bool operator==(const Entry& other) const;

private:
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_