#include "Entry.h"
#include "utility/Factory.h"
#include <string>
namespace sp
{

class EntryInterface
{
protected:
    Entry* m_self_;
    std::string m_name_;
    Entry* m_parent_;

public:
    EntryInterface(Entry* self = nullptr, const std::string& name = "", Entry* parent = nullptr);

    EntryInterface(const EntryInterface& other);

    EntryInterface(EntryInterface&& other);

    static std::unique_ptr<EntryInterface> create(const std::string& backend, Entry* self, const std::string& name = "", Entry* parent = nullptr);

    virtual ~EntryInterface() = default;

    virtual std::string prefix() const;

    virtual std::string name() const;

    virtual Entry::iterator parent() const;

    virtual Entry::Type type() const = 0;

    virtual EntryInterface* copy() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    virtual bool has_attribute(const std::string& name) const = 0;

    virtual Entry::single_t get_attribute_raw(const std::string& name) const = 0;

    virtual void set_attribute_raw(const std::string& name, const Entry::single_t& value) = 0;

    virtual void remove_attribute(const std::string& name) = 0;

    virtual std::map<std::string, Entry::single_t> attributes() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    virtual void set_single(const Entry::single_t&) = 0;
    virtual Entry::single_t get_single() const = 0;

    virtual void set_tensor(const Entry::tensor_t&) = 0;
    virtual Entry::tensor_t get_tensor() const = 0;

    virtual void set_block(const Entry::block_t&) = 0;
    virtual Entry::block_t get_block() const = 0;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0

    virtual Entry::iterator next() const = 0;

    // container
    virtual size_t size() const = 0;

    virtual Entry::range find(const Entry::pred_fun& pred) = 0;

    virtual void erase(const Entry::iterator& p) = 0;

    virtual void erase_if(const Entry::pred_fun& p) = 0;

    virtual void erase_if(const Entry::range& r, const Entry::pred_fun& p) = 0;

    // as array
    virtual Entry::iterator at(int idx) = 0;

    virtual Entry::iterator push_back() = 0;

    virtual Entry pop_back() = 0;
    virtual Range<Iterator<Entry>> items() const = 0;

    // as object

    virtual Entry::const_iterator find(const std::string& name) const = 0;

    virtual Entry::iterator find(const std::string& name) = 0;

    virtual Entry::iterator insert(const std::string& name) = 0;

    virtual Entry erase(const std::string& name) = 0;

    virtual Range<Iterator<const std::pair<const std::string, Entry>>> children() const = 0;
};

#define SP_REGISTER_ENTRY(_CLASS_NAME_)          \
    static bool REGISTERED_Entry##_CLASS_NAME_ = \
        ::sp::Factory<EntryInterface, Entry*, const std::string&, Entry*>::register_creator<Entry##_CLASS_NAME_>(__STRING(_CLASS_NAME_));

} // namespace sp