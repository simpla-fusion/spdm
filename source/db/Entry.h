#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
#include "../utility/Path.h"
#include "../utility/TypeTraits.h"
#include "HierarchicalTree.h"
#include <array>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
namespace sp
{
namespace db
{

class Entry;
class EntryCursor;
class EntryArray;

class EntryCursor
{
public:
    virtual ~EntryCursor() = default;

    virtual void next() const = 0; // traversal

    virtual bool same_as(const EntryCursor*) const = 0; // check

    // as tree node

    virtual std::shared_ptr<Entry> parent() const = 0;

    virtual std::shared_ptr<Entry> get() const = 0;
};

class Entry
{
public:
    typedef traits::pre_tagged_types element_type;

    Entry() = default;

    Entry(const Entry& other) = default;

    Entry(Entry&& other) = default;

    virtual ~Entry() = default;

    static std::unique_ptr<Entry> create(const std::string& request = "");

    static bool add_creator(const std::string& c_id, const std::function<Entry*()>&);

    virtual std::unique_ptr<Entry> copy() const { return nullptr; }

    //----------------------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_value(const element_type&) {}

    virtual element_type get_value() const { return nullptr; }

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    virtual size_t size() const { return 0; }

    virtual void clear() {}

    // as object

    virtual std::size_t count(const std::string& name) { return 0; }

    virtual std::unique_ptr<EntryCursor> insert(const std::string& path) { return nullptr; }

    virtual std::unique_ptr<EntryCursor> insert(const Path& path) { return nullptr; }

    virtual std::unique_ptr<EntryCursor> find(const std::string& path) { return nullptr; }

    virtual std::unique_ptr<EntryCursor> find(const Path& path) { return nullptr; }

    virtual std::unique_ptr<const EntryCursor> find(const std::string& path) const { return nullptr; }

    virtual std::unique_ptr<const EntryCursor> find(const Path& path) const { return nullptr; }

    virtual void erase(const std::string& path) {}

    virtual void erase(const Path& path) {}

    virtual std::unique_ptr<EntryCursor> first_child() { return nullptr; }

    virtual std::unique_ptr<const EntryCursor> first_child() const { return nullptr; }

    // level 1

    virtual std::unique_ptr<EntryCursor> select(const std::string& path) { return nullptr; }

    virtual std::unique_ptr<EntryCursor> select(const Path& path) { return nullptr; }

    virtual std::unique_ptr<const EntryCursor> select(const std::string& path) const { return nullptr; }

    virtual std::unique_ptr<const EntryCursor> select(const Path& path) const { return nullptr; }
};

class EntryArray
{
public:
    virtual ~EntryArray() = default;

    virtual std::unique_ptr<EntryArray> copy() const { return nullptr; };

    // as array
    virtual size_t size() const { return 0; }

    virtual void resize(std::size_t num){};

    virtual void clear(){};

    virtual std::unique_ptr<EntryCursor> push_back() { return nullptr; };

    virtual std::unique_ptr<EntryCursor> pop_back() { return nullptr; };

    virtual std::shared_ptr<const Entry> at(int idx) const { return nullptr; };

    virtual std::shared_ptr<Entry> at(int idx) { return nullptr; };
};

std::string to_string(Entry::element_type const& s);

Entry::element_type from_string(const std::string& s, int idx = 0);

} // namespace db
} // namespace sp

M_REGISITER_TYPE_TAG(Object, sp::db::Entry);
M_REGISITER_TYPE_TAG(Array, sp::db::EntryArray);
#endif //SP_ENTRY_H_