#ifndef SP_ENTRY_PLUGIN_H_
#define SP_ENTRY_PLUGIN_H_
#include "Entry.h"
#include <any>
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
template <typename Impl>
class EntryPlugin : public Entry
{

public:
    typedef EntryPlugin<Impl> this_type;

    template <typename... Args>
    EntryPlugin(Args&&... args) : m_pimpl_(std::forward<Args>(args)...) {}

    EntryPlugin(const std::string& request) : EntryPlugin(request){};

    EntryPlugin(const EntryPlugin& other) : EntryPlugin(other.m_pimpl_){};

    EntryPlugin(EntryPlugin&& other) : EntryPlugin(std::move(other.m_pimpl_)){};

    ~EntryPlugin() = default;

     std::unique_ptr<Entry> copy() const = 0;

    //  void init(const Attributes& ) = 0;

    //----------------------------------------------------------------------------------------------------------

     std::size_t type() const = 0;

     std::string path() const = 0;

     std::string name() const = 0;

    //  std::string name() const = 0;
    //----------------------------------------------------------------------------------------------------------
    // attribute
     bool has_attribute(const std::string& name) const = 0;

     element_t get_attribute_raw(const std::string& name) const = 0;

     void set_attribute_raw(const std::string& name, const element_t& value) = 0;

     void remove_attribute(const std::string& name) = 0;

     std::map<std::string, element_t> attributes() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
     void set_element(const element_t&) = 0;
     element_t get_element() const = 0;

     void set_tensor(const tensor_t&) = 0;
     tensor_t get_tensor() const = 0;

     void set_block(const block_t&) = 0;
     block_t get_block() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    //as cursor

     size_t size() const = 0;

     std::shared_ptr<Entry> next() const = 0; // traversal

     bool same_as(const Entry*) const = 0; // check

     void clear() = 0;

    // as tree node

     std::shared_ptr<Entry> parent() const = 0;

     std::shared_ptr<Entry> first_child() const = 0;

    // as array

     void resize(std::size_t num) = 0;

     std::shared_ptr<Entry> push_back() = 0;

     std::shared_ptr<Entry> pop_back() = 0;

     std::shared_ptr<const Entry> at(int idx) const = 0;

     std::shared_ptr<Entry> at(int idx) = 0;

    // as object

     std::size_t count(const std::string& name) = 0;

     std::shared_ptr<Entry> insert(const std::string& path) = 0;

     std::shared_ptr<Entry> insert(const Path& path) = 0;

     std::shared_ptr<Entry> find(const std::string& path) const = 0;

     std::shared_ptr<Entry> find(const Path& path) const = 0;

     void erase(const std::string& path) = 0;

     void erase(const Path& path) = 0;

    // level 1

     std::shared_ptr<Entry> select(const std::string& path) const { return nullptr; };

     std::shared_ptr<Entry> select(const Path& path) const { return nullptr; };

private:
    Impl m_pimpl_;
    static bool is_registered;
};

#define SP_REGISTER_ENTRY(_NAME_, _CLASS_)         \
    template <>                                    \
    bool sp::EntryPlugin<_CLASS_>::is_registered = \
        Entry::add_creator(                        \
            __STRING(_NAME_),                      \
            []() { return dynamic_cast<Entry*>(new EntryPlugin<_CLASS_>()); });

} // namespace sp
#endif // SP_ENTRY_PLUGIN_H_