#ifndef SPDB_ENTRY_H_
#define SPDB_ENTRY_H_
#include "HierarchicalTree.h"
#include "../utility/Path.h"
#include "../utility/TypeTraits.h"
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

    virtual void clear() = 0;

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

    virtual std::unique_ptr<Entry> copy() const = 0;

    //----------------------------------------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_value(const element_type&) = 0;

    virtual element_type get_value() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    // as object

    virtual std::size_t count(const std::string& name) = 0;

    virtual std::unique_ptr<EntryCursor> insert(const std::string& path) = 0;

    virtual std::unique_ptr<EntryCursor> insert(const Path& path) = 0;

    virtual std::unique_ptr<EntryCursor> find(const std::string& path) const = 0;

    virtual std::unique_ptr<EntryCursor> find(const Path& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual void erase(const Path& path) = 0;

    virtual std::unique_ptr<EntryCursor> first_child() const = 0;

    // level 1

    virtual std::unique_ptr<EntryCursor> select(const std::string& path) const = 0;

    virtual std::unique_ptr<EntryCursor> select(const Path& path) const = 0;
};

class EntryArray
{
public:
    virtual std::unique_ptr<EntryArray> copy() const = 0;

    // as array
    virtual size_t size() const = 0;

    virtual void resize(std::size_t num) = 0;

    virtual void clear() = 0;

    virtual std::shared_ptr<Entry> push_back() = 0;

    virtual std::shared_ptr<Entry> pop_back() = 0;

    virtual std::shared_ptr<const Entry> at(int idx) const = 0;

    virtual std::shared_ptr<Entry> at(int idx) = 0;
};

std::string to_string(Entry::element_type const& s);

Entry::element_type from_string(const std::string& s, int idx = 0);
} // namespace db
} // namespace sp

M_REGISITER_TYPE_TAG(Object, sp::db::Entry);
M_REGISITER_TYPE_TAG(Array, sp::db::EntryArray);
#endif //SP_ENTRY_H_