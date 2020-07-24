#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
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

class Entry : public std::enable_shared_from_this<Entry>
{
public:
    enum Type
    {
        Null = 0,
        Element = 1,
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

    Entry() = default;

    Entry(const Entry& other) = default;

    Entry(Entry&& other) = default;

    virtual ~Entry() = default;

    static std::unique_ptr<Entry> create(const std::string& rpath = "");

    static bool add_creator(const std::string& c_id, const std::function<Entry*()>&);

    virtual std::shared_ptr<Entry> copy() const = 0;

    //----------------------------------------------------------------------------------------------------------

    virtual Type type() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    virtual bool has_attribute(const std::string& name) const = 0;

    virtual element_t get_attribute_raw(const std::string& name) const = 0;

    virtual void set_attribute_raw(const std::string& name, const element_t& value) = 0;

    virtual void remove_attribute(const std::string& name) = 0;

    virtual std::map<std::string, element_t> attributes() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_element(const element_t&) = 0;
    virtual element_t get_element() const = 0;

    virtual void set_tensor(const tensor_t&) = 0;
    virtual tensor_t get_tensor() const = 0;

    virtual void set_block(const block_t&) = 0;
    virtual block_t get_block() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    //as cursor

    virtual size_t size() const = 0;

    virtual std::shared_ptr<Entry> next() const = 0; // traversal

    virtual bool same_as(const Entry*) const = 0; // check

    // as tree node

    virtual std::shared_ptr<Entry> parent() const = 0;

    virtual std::shared_ptr<Entry> first_child() const = 0;

    // as array

    virtual std::shared_ptr<Entry> push_back() = 0;

    virtual std::shared_ptr<Entry> pop_back() = 0;

    virtual std::shared_ptr<Entry> item(int idx) const = 0;

    // as object
    virtual std::shared_ptr<Entry> insert(const std::string& path) = 0;

    virtual std::shared_ptr<Entry> insert_r(const std::string& path) = 0;

    virtual std::shared_ptr<Entry> find(const std::string& path) const = 0;

    virtual std::shared_ptr<Entry> find_r(const std::string& path) const = 0;

    virtual void remove(const std::string& path) = 0;

    // level 1

    virtual std::shared_ptr<Entry> select(const std::string& path) const { return nullptr; };

    // virtual std::shared_ptr<Entry::iterator> select(const XPath& path) const { return nullptr; };
};
} // namespace sp

#endif //SP_ENTRY_H_