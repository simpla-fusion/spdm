
#include "Entry.h"
#include "EntryInterface.h"
#include "utility/Logger.h"
#include <variant>
namespace sp
{

class EntryMemory : public EntryInterface
{
private:
    std::variant<nullptr_t,
                 Entry::single_t,
                 Entry::tensor_t,
                 Entry::block_t,
                 std::vector<Entry>,
                 std::map<std::string, Entry>>
        m_data_;

public:
    using EntryInterface::m_name_;
    using EntryInterface::m_parent_;
    using EntryInterface::m_self_;

    EntryMemory(Entry* self, const std::string& name = "", Entry* parent = nullptr)
        : EntryInterface(self, name, parent),
          m_data_(nullptr){};

    EntryMemory(const EntryMemory& other)
        : EntryInterface(other), m_data_(other.m_data_) {}

    EntryMemory(EntryMemory&& other)
        : EntryInterface(std::forward<EntryMemory>(other)), m_data_(std::move(other.m_data_)) {}

    ~EntryMemory() = default;

    EntryInterface* copy() const
    {
        return new EntryMemory(*this);
    };

    Entry::Type type() const { return Entry::Type(m_data_.index()); }

    // attributes

    bool has_attribute(const std::string& name) const { return !find("@" + name); }

    Entry::single_t get_attribute_raw(const std::string& name) const
    {
        auto p = find("@" + name);
        if (!p)
        {
            throw std::out_of_range(FILE_LINE_STAMP_STRING + "Can not find attribute '" + name + "'");
        }
        return p->get_single();
    }

    void set_attribute_raw(const std::string& name, const Entry::single_t& value) { insert("@" + name)->set_single(value); }

    void remove_attribute(const std::string& name) { erase("@" + name); }

    std::map<std::string, Entry::single_t> attributes() const
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

    void set_single(const Entry::single_t& v) override
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

    Entry::single_t get_single() const override
    {
        if (type() != Entry::Type::Single)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not Single!");
        }
        return std::get<Entry::Type::Single>(m_data_);
    }

    void set_tensor(const Entry::tensor_t& v) override
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

    Entry::tensor_t get_tensor() const override
    {
        if (type() != Entry::Type::Tensor)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
        }
        return std::get<Entry::Type::Tensor>(m_data_);
    }

    void set_block(const Entry::block_t& v) override
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

    Entry::block_t get_block() const override
    {
        if (type() != Entry::Type::Block)
        {
            throw std::runtime_error(FILE_LINE_STAMP_STRING + "This is not block!");
        }
        return std::get<Entry::Type::Block>(m_data_);
    }

    // as Tree

    // Entry::iterator parent() const override { return Entry::iterator(const_cast<Entry*>(m_parent_)); }

    Entry::iterator next() const override
    {
        NOT_IMPLEMENTED;
        return Entry::iterator();
    };

    Range<Iterator<Entry>> items() const override
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

    Range<Iterator<const std::pair<const std::string, Entry>>> children() const
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

    size_t size() const
    {
        NOT_IMPLEMENTED;
        return 0;
    }

    Entry::range find(const Entry::pred_fun& pred)
    {
        NOT_IMPLEMENTED;
    }

    void erase(const Entry::iterator& p)
    {
        NOT_IMPLEMENTED;
    }

    void erase_if(const Entry::pred_fun& p)
    {
        NOT_IMPLEMENTED;
    }

    void erase_if(const Entry::range& r, const Entry::pred_fun& p)
    {
        NOT_IMPLEMENTED;
    }

    // as vector
    Entry::iterator at(int idx)
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

    Entry::iterator push_back()
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

    Entry pop_back()
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

    // as object
    Entry::const_iterator find(const std::string& name) const
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

    Entry::iterator find(const std::string& name)
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

    Entry::iterator insert(const std::string& name)
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

    Entry erase(const std::string& name)
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
};
SP_REGISTER_ENTRY(Memory)
} // namespace sp