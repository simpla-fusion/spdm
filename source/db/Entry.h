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

    Entry(const Node& opt);

    Entry(std::initializer_list<Node> init, Path p = {});

    Entry(const std::shared_ptr<NodeObject>& root, Path p = {});

    Entry(const Entry& other);

    Entry(Entry&& other);

    ~Entry() = default;

    void swap(Entry& other);

    Entry& operator=(const Entry& other);

    static Entry create(const Node&);

    void load(const Node&);

    void save(const Node&) const;

    std::shared_ptr<NodeObject> root();

    const std::shared_ptr<NodeObject> root() const;

    Path& path() { return m_path_; }

    const Path& path() const { return m_path_; }

    void reset();

    //-------------------------------------------------------------------------

    void update(Node const& v);

    Node fetch(Node const& projection = {});

    Node fetch(Node const& projection = {}) const;

    //-------------------------------------------------------------------------

    bool operator==(const Entry& other) const { return same_as(other); }

    bool same_as(const Entry& other) const;

    size_t type() const;

    bool is_null() const;

    bool empty() const;

    size_t count() const;

    template <typename V, typename... Others,
              std::enable_if_t<sizeof...(Others) != 0 && std::is_constructible_v<Node, Others...>, int> = 0>
    void set_value(Others&&... others) { update(Node(std::in_place_type_t<V>(), std::forward<Others>(others)...)); }

    template <int IDX, typename... Others,
              std::enable_if_t<sizeof...(Others) != 0 && std::is_constructible_v<Node, Others...>, int> = 0>
    void set_value(Others&&... others) { update(Node(std::in_place_index_t<IDX>(), std::forward<Others>(others)...)); }

    template <typename V, typename... Args>
    V get_value(Args&&... args) const { return fetch().get_value<V>(std::forward<Args>(args)...); }

    template <int IDX, typename... Args>
    auto get_value(Args&&... args) const { return fetch().get_value<IDX>(std::forward<Args>(args)...); }

    template <typename V>
    Entry& operator=(const V& v)
    {
        set_value<V>(v);
        return *this;
    }

    Entry& operator=(const char* v)
    {
        set_value<std::string>(v);
        return *this;
    }

    //-------------------------------------------------------------------------
    // access
    template <typename... Args>
    Entry at(Args&&... args) & { return Entry(root(), Path(m_path_).join(std::forward<Args>(args)...)); }

    template <typename... Args>
    Entry at(Args&&... args) && { return Entry(std::move(root()), Path(std::move(m_path_)).join(std::forward<Args>(args)...)); }

    template <typename... Args>
    Entry at(Args&&... args) const& { return Entry(root(), Path(m_path_).join(std::forward<Args>(args)...)); }

    template <typename T>
    Entry operator[](const T& idx) & { return at(idx); }
    template <typename T>
    Entry operator[](const T& idx) && { return at(idx); }
    template <typename T>
    Entry operator[](const T& idx) const& { return at(idx); }

    template <typename... Args>
    Entry slice(Args&&... args) & { return at(Slice(std::forward<Args>(args)...)); }
    template <typename... Args>
    Entry slice(Args&&... args) && { return at(Slice(std::forward<Args>(args)...)); }
    template <typename... Args>
    Entry slice(Args&&... args) const& { return at(Slice(std::forward<Args>(args)...)); }

    template <typename V>
    auto attribute(const std::string& name) const { return at("@" + name).get_value<V>(); }

    template <typename V, typename... Args>
    void attribute(const std::string& name, Args&&... args) { at("@" + name).set_value<V>(std::forward<Args>(args)...); }

    //-------------------------------------------------------------------------

    void resize(int num);

    Node pop_back();

    Entry push_back(Node v = {});

    Cursor<Node> children();

    Cursor<const Node> children() const;

private:
};

std::ostream& operator<<(std::ostream& os, Entry const& entry);

} // namespace sp::db

#endif //SP_ENTRY_H_