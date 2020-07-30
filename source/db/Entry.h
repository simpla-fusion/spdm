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

using element_types = std::variant<
    std::tuple<std::shared_ptr<void>, int, std::vector<size_t>>, //Block
    std::string,                                                 //String,
    bool,                                                        //Boolean,
    int,                                                         //Integer,
    long,                                                        //Long,
    float,                                                       //Float,
    double,                                                      //Double,
    std::complex<double>,                                        //Complex,
    std::array<int, 3>,                                          //IntVec3,
    std::array<long, 3>,                                         //LongVec3,
    std::array<float, 3>,                                        //FloatVec3,
    std::array<double, 3>,                                       //DoubleVec3,
    std::array<std::complex<double>, 3>,                         //ComplexVec3,
    std::any>;

class EntryArray;

class Entry
{
public:
    typedef traits::concatenate_t<std::variant<std::nullptr_t, std::shared_ptr<Entry>, std::shared_ptr<EntryArray>>, element_types> element;

    typedef traits::type_tags<element> type_tags;

    typedef Cursor<element> cursor;

    typedef Cursor<const element> const_cursor;

    Entry() = default;

    Entry(const Entry& other) = default;

    Entry(Entry&& other) = default;

    virtual ~Entry() = default;

    static std::shared_ptr<Entry> create(const std::string& request = "");

    static bool add_creator(const std::string& c_id, const std::function<Entry*()>&);

    virtual std::shared_ptr<Entry> copy() const = 0;

    //----------------------------------------------------------------------------------------------------------
    virtual std::size_t type(const std::string& path) const;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_value(const std::string& path, const element&) = 0;

    virtual element get_value(const std::string& path) const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    virtual size_t size() const = 0;

    virtual void clear() = 0;

    // as object

    virtual std::size_t count(const std::string& name) = 0;

    virtual cursor insert(const std::string& path) = 0;

    virtual cursor insert(const Path& path) = 0;

    virtual cursor find(const std::string& path) = 0;

    virtual cursor find(const Path& path) = 0;

    virtual const_cursor find(const std::string& path) const = 0;

    virtual const_cursor find(const Path& path) const = 0;

    virtual void erase(const std::string& path) = 0;

    virtual void erase(const Path& path) = 0;

    virtual cursor first_child() = 0;

    virtual const_cursor first_child() const = 0;

    // level 1

    virtual cursor select(const std::string& path) = 0;

    virtual cursor select(const Path& path) = 0;

    virtual const_cursor select(const std::string& path) const = 0;

    virtual const_cursor select(const Path& path) const = 0;
};

class EntryArray
{
public:
    EntryArray();
    EntryArray(const EntryArray&);
    EntryArray(EntryArray&&);

    virtual ~EntryArray() = default;

    virtual std::shared_ptr<EntryArray> copy() const = 0;

    virtual size_t size() const = 0;

    virtual void resize(std::size_t num) = 0;

    virtual void clear() = 0;

    virtual Entry::cursor push_back() = 0;

    virtual void pop_back() = 0;

    virtual Entry::element& at(int idx) = 0;

    virtual const Entry::element& at(int idx) const = 0;
};

std::string to_string(Entry::element const& s);

Entry::element from_string(const std::string& s, int idx = 0);

template <typename TNode>
using entry_wrapper = traits::template_copy_type_args<
    HierarchicalTree,
    traits::concatenate_t<
        std::variant<TNode,
                     Entry,
                     EntryArray>,
        element_types>>;

} // namespace db
} // namespace sp

M_REGISITER_TYPE_TAG(Object, sp::db::Entry);
M_REGISITER_TYPE_TAG(Array, sp::db::EntryArray);
#endif //SP_ENTRY_H_