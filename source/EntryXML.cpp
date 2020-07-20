#include "Entry.h"
#include "EntryInterface.h"
#include "utility/Logger.h"
#include <any>
#include <map>
#include <pugixml/pugixml.hpp>
#include <vector>

namespace sp
{

class EntryXML : public EntryInterface
{
public:
    EntryXML(Entry* self, const std::string& name = "", Entry* parent = nullptr);
    EntryXML(const EntryXML&);
    EntryXML(EntryXML&&);
    ~EntryXML();

    void swap(EntryXML& other);

    EntryInterface* copy() const override;

    std::string prefix() const override;

    std::string name() const override;

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
    Entry::iterator parent() const override;

    Entry::iterator next() const override;

    // container
    size_t size() const override;

    Entry::range find(const Entry::pred_fun& pred) override;

    void erase(const Entry::iterator& p) override;

    void erase_if(const Entry::pred_fun& p) override;

    void erase_if(const Entry::range& r, const Entry::pred_fun& p) override;

    // as array
    Entry::iterator at(int idx) override;

    Entry::iterator push_back() override;

    Entry pop_back() override;

    Range<Iterator<Entry>> items() const override;

    // as object

    Entry::const_iterator find(const std::string& name) const override;

    Entry::iterator find(const std::string& name) override;

    Entry::iterator insert(const std::string& name) override;

    Entry erase(const std::string& name) override;

    Range<Iterator<const std::pair<const std::string, Entry>>> children() const override;

private:
    Entry* m_self_;
    std::string m_name_;
    Entry* m_parent_;

    std::variant<nullptr_t,
                 Entry::single_t,
                 Entry::tensor_t,
                 Entry::block_t,
                 std::vector<Entry>,
                 std::map<std::string, Entry>>
        m_data_;
};
//
// class EntryXML;

EntryXML::EntryXML(Entry* self, const std::string& name, Entry* parent)
    : m_self_(self),
      m_name_(name),
      m_parent_(parent),
      m_data_(nullptr){};

EntryXML::EntryXML(const EntryXML& other)
    : m_self_(other.m_self_),
      m_name_(other.m_name_),
      m_parent_(other.m_parent_),
      m_data_(other.m_data_) {}

EntryXML::EntryXML(EntryXML&& other)
    : m_self_(other.m_self_),
      m_name_(other.m_name_),
      m_parent_(other.m_parent_),
      m_data_(std::move(other.m_data_)) {}

EntryXML::~EntryXML(){};

void EntryXML::swap(EntryXML& other)
{
}

EntryInterface* EntryXML::copy() const
{
    return new EntryXML(*this);
};

//

std::string EntryXML::prefix() const
{
    NOT_IMPLEMENTED;
    return "";
}

std::string EntryXML::name() const
{
    if (m_parent_ == nullptr)
    {
        return "";
    }
    else if (m_parent_->type() == Entry::Type::Array)
    {
        return m_parent_->name();
    }
    else
    {
        return m_name_;
    }
}

Entry::Type EntryXML::type() const
{
    return Entry::Type(m_data_.index());
}

// attributes

bool EntryXML::has_attribute(const std::string& name) const
{
    return !find("@" + name);
}

Entry::single_t EntryXML::get_attribute_raw(const std::string& name) const
{
    auto p = find("@" + name);
    if (!p)
    {
        throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
    }
    return p->get_single();
}

void EntryXML::set_attribute_raw(const std::string& name, const Entry::single_t& value)
{
    insert("@" + name)->set_single(value);
}

void EntryXML::remove_attribute(const std::string& name)
{
    erase("@" + name);
}

std::map<std::string, Entry::single_t> EntryXML::attributes() const
{
    if (type() != Entry::Type::Object)
    {
        return std::map<std::string, Entry::single_t>{};
    }

    std::map<std::string, Entry::single_t> res;
    for (const auto& item : std::get<Entry::Type::Object>(m_data_))
    {
        if (item.first[0] == '@')
        {
            res.emplace(item.first.substr(1, std::string::npos), item.second.get_single());
        }
    }
    return std::move(res);
}

//----------------------------------------------------------------------------------
// level 0
//
// as leaf

void EntryXML::set_single(const Entry::single_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_data_.emplace<Entry::Type::Single>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}

Entry::single_t EntryXML::get_single() const
{
    if (type() != Entry::Type::Single)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Single!");
    }
    return std::get<Entry::Type::Single>(m_data_);
}

void EntryXML::set_tensor(const Entry::tensor_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_data_.emplace<Entry::Type::Tensor>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}

Entry::tensor_t EntryXML::get_tensor() const
{
    if (type() != Entry::Type::Tensor)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Tensor>(m_data_);
}

void EntryXML::set_block(const Entry::block_t& v)
{
    if (type() < Entry::Type::Array)
    {
        m_data_.emplace<Entry::Type::Block>(v);
    }
    else
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "Set value failed!");
    }
}

Entry::block_t EntryXML::get_block() const
{
    if (type() != Entry::Type::Block)
    {
        throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
    }
    return std::get<Entry::Type::Block>(m_data_);
}

// as Tree

Entry::iterator EntryXML::parent() const
{
    return Entry::iterator(const_cast<Entry*>(m_parent_));
}

Entry::iterator EntryXML::next() const
{
    NOT_IMPLEMENTED;
    return Entry::iterator();
};

Range<Iterator<Entry>> EntryXML::items() const
{
    if (type() == Entry::Type::Array)
    {
        auto& m = std::get<Entry::Type::Array>(m_data_);
        return Entry::range{Entry::iterator(m.begin()),
                            Entry::iterator(m.end())};
        ;
    }
    // else if (type() == Entry::Type::Object)
    // {
    //     auto& m = std::get<Entry::Type::Object>(m_data_);
    //     auto mapper = [](auto const& item) -> Entry* { return &item->second; };
    //     return Entry::range{Entry::iterator(m.begin(), mapper),
    //                         Entry::iterator(m.end(), mapper)};
    // }

    return Entry::range{};
}

Range<Iterator<const std::pair<const std::string, Entry>>> EntryXML::children() const
{
    if (type() == Entry::Type::Object)
    {
        auto& m = std::get<Entry::Type::Object>(m_data_);

        return Range<Iterator<const std::pair<const std::string, Entry>>>{
            Iterator<const std::pair<const std::string, Entry>>(m.begin()),
            Iterator<const std::pair<const std::string, Entry>>(m.end())};
    }

    return Range<Iterator<const std::pair<const std::string, Entry>>>{};
}

size_t EntryXML::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}

Entry::range EntryXML::find(const Entry::pred_fun& pred)
{
    NOT_IMPLEMENTED;
}

void EntryXML::erase(const Entry::iterator& p)
{
    NOT_IMPLEMENTED;
}

void EntryXML::erase_if(const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}

void EntryXML::erase_if(const Entry::range& r, const Entry::pred_fun& p)
{
    NOT_IMPLEMENTED;
}

Entry::iterator EntryXML::at(int idx)
{
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_data_);
        return Entry::iterator(&m[idx]);
    }
    catch (std::bad_variant_access&)
    {
        return Entry::iterator();
    };
}

Entry::iterator EntryXML::push_back()
{
    if (type() == Entry::Type::Null)
    {
        m_data_.emplace<Entry::Type::Array>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_data_);
        m.emplace_back(Entry(m_self_));
        return Entry::iterator(&*m.rbegin());
    }
    catch (std::bad_variant_access&)
    {
        return Entry::iterator();
    };
}

Entry EntryXML::pop_back()
{
    try
    {
        auto& m = std::get<Entry::Type::Array>(m_data_);
        Entry res;
        m.rbegin()->swap(res);
        m.pop_back();
        return std::move(res);
    }
    catch (std::bad_variant_access&)
    {
        return Entry();
    }
}

Entry::const_iterator EntryXML::find(const std::string& name) const
{
    try
    {
        auto const& m = std::get<Entry::Type::Object>(m_data_);
        auto it = m.find(name);
        if (it != m.end())
        {
            return it->second.self();
        }
    }
    catch (std::bad_variant_access&)
    {
    }
    return Entry::const_iterator();
}

Entry::iterator EntryXML::find(const std::string& name)
{
    try
    {
        auto const& m = std::get<Entry::Type::Object>(m_data_);
        auto it = m.find(name);
        if (it != m.end())
        {
            return const_cast<Entry&>(it->second).self();
        }
    }
    catch (std::bad_variant_access&)
    {
    }
    return Entry::iterator();
}

Entry::iterator EntryXML::insert(const std::string& name)
{
    if (type() == Entry::Type::Null)
    {
        m_data_.emplace<Entry::Type::Object>();
    }
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_data_);

        return Entry::iterator(&(m.emplace(name, Entry(m_self_, name)).first->second));
    }
    catch (std::bad_variant_access&)
    {
        return Entry::iterator();
    }
}

Entry EntryXML::erase(const std::string& name)
{
    try
    {
        auto& m = std::get<Entry::Type::Object>(m_data_);
        auto it = m.find(name);
        if (it != m.end())
        {
            Entry res;
            res.swap(it->second);
            m.erase(it);
            return std::move(res);
        }
    }
    catch (std::bad_variant_access&)
    {
    }
    return Entry();
}

SP_REGISTER_ENTRY(XML);
} // namespace sp
