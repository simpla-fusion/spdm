#include "Entry.h"
#include <string>
namespace sp
{

class EntryInterface
{
protected:
    Entry* m_self_;

public:
    EntryInterface();

    EntryInterface(const EntryInterface& other);

    EntryInterface(EntryInterface&& other);

    static std::unique_ptr<EntryInterface> create(const std::string& rpath = "");

    static bool add_creator(const std::string& c_id, const std::function<EntryInterface*()>&);

    virtual ~EntryInterface() = default;

    void bind(Entry* self);

    Entry* self();

    const Entry* self() const;

    virtual Entry::Type type() const = 0;

    virtual EntryInterface* copy() const = 0;

    virtual EntryInterface* duplicate() const = 0;

    virtual int fetch(const std::string& uri) = 0;

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
    virtual std::shared_ptr<Entry> at(int idx) = 0;

    virtual std::shared_ptr<Entry> push_back() = 0;

    virtual std::shared_ptr<Entry> pop_back() = 0;

    virtual Entry::range items() const = 0;

    // as object

    virtual const std::shared_ptr<Entry> find(const std::string& name) const = 0;

    virtual std::shared_ptr<Entry> find(const std::string& name) = 0;

    virtual std::shared_ptr<Entry> insert(const std::string& name) = 0;

    virtual std::shared_ptr<Entry> erase(const std::string& name) = 0;

    virtual Range<Iterator<const std::pair<const std::string, std::shared_ptr<Entry>>>> children() const = 0;
};

template <typename Impl>
class EntryImplement : public EntryInterface
{

public:
    using EntryInterface::m_self_;

    EntryImplement();
    EntryImplement(Impl const&);
    EntryImplement(const EntryImplement&);
    EntryImplement(EntryImplement&&);
    ~EntryImplement();

    void swap(EntryImplement& other);

    EntryInterface* copy() const override;
    EntryInterface* duplicate() const override;

    int fetch(const std::string& uri);

    Entry::Type type() const override;
    //----------------------------------------------------------------------------------------------------------
    // attribute
    //----------------------------------------------------------------------------------------------------------
    bool has_attribute(const std::string& name) const override;

    Entry::single_t get_attribute_raw(const std::string& name) const override;

    void set_attribute_raw(const std::string& name, const Entry::single_t& value) override;

    void remove_attribute(const std::string& name) override;

    std::map<std::string, Entry::single_t> attributes() const override;

    //----------------------------------------------------------------------------------------------------------
    // as leaf node,  need node.type = Scalar || Block
    //----------------------------------------------------------------------------------------------------------
    void set_single(const Entry::single_t&) override;
    Entry::single_t get_single() const override;

    void set_tensor(const Entry::tensor_t&) override;
    Entry::tensor_t get_tensor() const override;

    void set_block(const Entry::block_t&) override;
    Entry::block_t get_block() const override;

    //----------------------------------------------------------------------------------------------------------
    // as Hierarchy tree node
    // function level 0
    // Entry::iterator parent() const override;

    Entry::iterator next() const override;

    // container
    size_t size() const override;

    Entry::range find(const Entry::pred_fun& pred) override;

    void erase(const Entry::iterator& p) override;

    void erase_if(const Entry::pred_fun& p) override;

    void erase_if(const Entry::range& r, const Entry::pred_fun& p) override;

    // as array
    std::shared_ptr<Entry> at(int idx) override;

    std::shared_ptr<Entry> push_back() override;

    std::shared_ptr<Entry> pop_back() override;

    Entry::range items() const override;

    // as object

    const std::shared_ptr<Entry> find(const std::string& name) const override;

    std::shared_ptr<Entry> find(const std::string& name) override;

    std::shared_ptr<Entry> insert(const std::string& name) override;

    std::shared_ptr<Entry> erase(const std::string& name) override;

    Range<Iterator<const std::pair<const std::string, std::shared_ptr<Entry>>>> children() const override;

private:
    Impl m_pimpl_;
    static bool is_registered;
};

#define SP_REGISTER_ENTRY(_NAME_, _CLASS_)            \
    template <>                                       \
    bool sp::EntryImplement<_CLASS_>::is_registered = \
        EntryInterface::add_creator(                  \
            __STRING(_NAME_),                         \
            []() { return dynamic_cast<EntryInterface*>(new EntryImplement<_CLASS_>()); });

} // namespace sp