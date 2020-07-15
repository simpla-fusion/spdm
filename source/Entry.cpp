#include "Entry.h"
#include "utility/Logger.h"
namespace sp
{
struct entry_in_memory
{
    entry_in_memory* create() const { return new entry_in_memory(); }
    entry_in_memory* copy() const { return new entry_in_memory{*this}; }
    TypeTag type_tag() const { return m_type_tag_; }

    TypeTag m_type_tag_;
};

template <>
void EntryPolicyBase<entry_in_memory>::resolve() { NOT_IMPLEMENTED; }

//----------------------------------------------------------------
// attributes
template <>
bool EntryPolicyBase<entry_in_memory>::has_attribute(std::string const& key) const
{
    NOT_IMPLEMENTED;
    return false;
}
template <>
bool EntryPolicyBase<entry_in_memory>::check_attribute(std::string const& key, std::any const& v) const
{
    NOT_IMPLEMENTED;
    return false;
}
template <>
void EntryPolicyBase<entry_in_memory>::set_attribute(const std::string&, const std::any&)
{
    NOT_IMPLEMENTED;
}
template <>
std::any EntryPolicyBase<entry_in_memory>::get_attribute(const std::string&) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::any EntryPolicyBase<entry_in_memory>::get_attribute(std::string const& key, std::any const& default_value)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
void EntryPolicyBase<entry_in_memory>::remove_attribute(const std::string&)
{
    NOT_IMPLEMENTED;
}
template <>
Range<Iterator<std::pair<std::string, std::any>>>
EntryPolicyBase<entry_in_memory>::attributes() const
{
    NOT_IMPLEMENTED;
    return Range<Iterator<std::pair<std::string, std::any>>>{};
}
template <>
void EntryPolicyBase<entry_in_memory>::clear_attributes()
{
    NOT_IMPLEMENTED;
}

//-------------------------------------------------------------------------------------------------------
// as scalar
template <>
std::any EntryPolicy<entry_in_memory, TypeTag::Scalar>::get_scalar() const { return nullptr; }

template <>
void EntryPolicy<entry_in_memory, TypeTag::Scalar>::set_scalar(std::any const&) {}

//-------------------------------------------------------------------------------------------------------
// as block

template <>
std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>
EntryPolicy<entry_in_memory, TypeTag::Block>::get_raw_block() const
{
    NOT_IMPLEMENTED;
    return std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>{nullptr, typeid(nullptr_t), {}};
}
template <>
void EntryPolicy<entry_in_memory, TypeTag::Block>::set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                                                                 const std::type_info& /*element type*/,
                                                                 const std::vector<size_t>& /*dimensions*/)
{
    NOT_IMPLEMENTED;
}
//-------------------------------------------------------------------------------------------------------
// array
template <>
size_t EntryPolicy<entry_in_memory, TypeTag::Array>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Entry::range EntryPolicy<entry_in_memory, TypeTag::Array>::children()
{
    NOT_IMPLEMENTED;
    return Entry::range{};
}
template <>
Entry::const_range EntryPolicy<entry_in_memory, TypeTag::Array>::children() const
{
    NOT_IMPLEMENTED;
    return Entry::const_range{};
}
template <>
void EntryPolicy<entry_in_memory, TypeTag::Array>::clear_children() { NOT_IMPLEMENTED; }
template <>
void EntryPolicy<entry_in_memory, TypeTag::Array>::remove_child(Entry::iterator const&) { NOT_IMPLEMENTED; }
template <>
void EntryPolicy<entry_in_memory, TypeTag::Array>::remove_children(Entry::range const&) { NOT_IMPLEMENTED; }
template <>
Entry::iterator EntryPolicy<entry_in_memory, TypeTag::Array>::begin()
{
    NOT_IMPLEMENTED;
    return Entry::iterator{};
}
template <>
Entry::iterator EntryPolicy<entry_in_memory, TypeTag::Array>::end()
{
    NOT_IMPLEMENTED;
    return Entry::iterator{};
}
template <>
Entry::const_iterator EntryPolicy<entry_in_memory, TypeTag::Array>::cbegin() const
{
    NOT_IMPLEMENTED;
    return Entry::const_iterator{};
}
template <>
Entry::const_iterator EntryPolicy<entry_in_memory, TypeTag::Array>::cend() const
{
    NOT_IMPLEMENTED;
    return Entry::const_iterator{};
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::push_back(const std::shared_ptr<Entry>& p)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::push_back(Entry&&)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::push_back(const Entry&)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry::range EntryPolicy<entry_in_memory, TypeTag::Array>::push_back(const Entry::iterator& b, const Entry::iterator& e)
{
    NOT_IMPLEMENTED;
    return Entry::range{};
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::at(int idx)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::at(int idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::find_child(size_t)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Entry> EntryPolicy<entry_in_memory, TypeTag::Array>::find_child(size_t) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

//---------------------------------------------------------------------------------------------------
// Table
template <>
size_t EntryPolicy<entry_in_memory, TypeTag::Table>::size() const
{
    NOT_IMPLEMENTED;
    return 0;
}
template <>
Entry::range EntryPolicy<entry_in_memory, TypeTag::Table>::children()
{
    NOT_IMPLEMENTED;
    return Entry::range{};
}
template <>
Entry::const_range EntryPolicy<entry_in_memory, TypeTag::Table>::children() const
{
    NOT_IMPLEMENTED;
    return Entry::const_range{};
}
template <>
void EntryPolicy<entry_in_memory, TypeTag::Table>::clear_children()
{
    NOT_IMPLEMENTED;
}
template <>
void EntryPolicy<entry_in_memory, TypeTag::Table>::remove_child(Entry::iterator const&)
{
    NOT_IMPLEMENTED;
}
template <>
void EntryPolicy<entry_in_memory, TypeTag::Table>::remove_children(Entry::range const&)
{
    NOT_IMPLEMENTED;
}
template <>
Entry::iterator EntryPolicy<entry_in_memory, TypeTag::Table>::begin()
{
    NOT_IMPLEMENTED;
    return Entry::iterator{};
}
template <>
Entry::iterator EntryPolicy<entry_in_memory, TypeTag::Table>::end()
{
    NOT_IMPLEMENTED;
    return Entry::iterator{};
}
template <>
Entry::const_iterator EntryPolicy<entry_in_memory, TypeTag::Table>::cbegin() const
{
    NOT_IMPLEMENTED;
    return Entry::const_iterator{};
}
template <>
Entry::const_iterator EntryPolicy<entry_in_memory, TypeTag::Table>::cend() const
{
    NOT_IMPLEMENTED;
    return Entry::const_iterator{};
}
template <>
Entry::const_range_kv EntryPolicy<entry_in_memory, TypeTag::Table>::items() const
{
    NOT_IMPLEMENTED;
    return Entry::const_range_kv{};
}
template <>
Entry::range_kv EntryPolicy<entry_in_memory, TypeTag::Table>::items()
{
    NOT_IMPLEMENTED;
    return Entry::range_kv{};
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Table>::insert(const std::string& k, std::shared_ptr<Entry> const& node)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
Entry::range_kv EntryPolicy<entry_in_memory, TypeTag::Table>::insert(const Entry::iterator_kv& b, const Entry::iterator_kv& e)
{
    NOT_IMPLEMENTED;
    return Entry::range_kv{};
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Table>::at(const std::string& key)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Entry> EntryPolicy<entry_in_memory, TypeTag::Table>::at(const std::string& idx) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<Entry> EntryPolicy<entry_in_memory, TypeTag::Table>::find_child(const std::string&)
{
    NOT_IMPLEMENTED;
    return nullptr;
}
template <>
std::shared_ptr<const Entry> EntryPolicy<entry_in_memory, TypeTag::Table>::find_child(const std::string&) const
{
    NOT_IMPLEMENTED;
    return nullptr;
}

// //-----------------------------------------------------------------------------------------------------
// // content

// struct Content
// {
//     virtual bool is_leaf() const { return true; }
//     virtual std::type_info const& type_info() const { return typeid(Content); };
//     virtual Content* copy() const { return nullptr; }
//     virtual Content* move() { return nullptr; }
// };

// struct ContentScalar : public Content
// {
//     std::any content;
//     ContentScalar() {}
//     ContentScalar(const ContentScalar& other) : content(other.content) {}
//     ContentScalar(ContentScalar&& other) : content(std::move(other.content)) {}
//     std::type_info const& type_info() const { return typeid(ContentScalar); }
//     Content* copy() const override { return new ContentScalar(*this); };
//     Content* move() override { return new ContentScalar(std::move(*this)); };
// };

// struct ContentBlock : public Content
// {
//     std::tuple<std::shared_ptr<char>, std::type_info const&, std::vector<size_t>> content;
//     ContentBlock() : content(nullptr, typeid(nullptr_t), {}) {}
//     ContentBlock(std::shared_ptr<char> const& p, std::type_info const& t, std::vector<size_t> const& d) : content({p, t, d}) {}
//     ContentBlock(const ContentBlock& other) : content(other.content) {}
//     ContentBlock(ContentBlock&& other) : content(std::move(other.content)) {}
//     std::type_info const& type_info() const { return typeid(ContentBlock); }
//     Content* copy() const { return new ContentBlock{*this}; };
//     Content* move() { return new ContentBlock{std::move(*this)}; };
// };

// struct ContentTree : public Content
// {
//     bool is_leaf() const override { return false; }
// };
// struct ContentArray : public ContentTree
// {
//     std::vector<std::shared_ptr<Entry>> content;
//     ContentArray() {}
//     ContentArray(ContentArray const& other) : content(other.content) {}
//     ContentArray(ContentArray&& other) : content(std::move(other.content)) {}
//     std::type_info const& type_info() const { return typeid(ContentArray); }
//     Content* copy() const { return new ContentArray{*this}; };
//     Content* move() { return new ContentArray{std::move(*this)}; };
// };

// struct ContentTable : public Content
// {
//     std::map<std::string, std::shared_ptr<Entry>> content;
//     ContentTable() {}
//     ContentTable(ContentTable const& other) : content(other.content) {}
//     ContentTable(ContentTable&& other) : content(std::move(other.content)) {}
//     std::type_info const& type_info() const { return typeid(ContentTable); }
//     Content* copy() const { return new ContentTable{*this}; };
//     Content* move() { return new ContentTable{std::move(*this)}; };
// };

// EntryInMemory::EntryInMemory()
//     : m_attributes_(new Attributes),
//       m_content_(new ContentScalar) {}

// EntryInMemory::EntryInMemory(EntryInMemory const& other)
//     : m_attributes_(other.m_attributes_->copy()),
//       m_content_(other.m_content_->copy()) {}

// EntryInMemory::EntryInMemory(EntryInMemory&& other)
//     : m_attributes_(other.m_attributes_.release()),
//       m_content_(other.m_content_.release()) {}

// EntryInMemory::~EntryInMemory() {}

// //------------------------------------------------------------------------------------
// // attribute

// bool EntryInMemory::has_attribute(std::string const& key) const { return m_attributes_->has_a(key); }

// bool EntryInMemory::check_attribute(std::string const& key, std::any const& v) const { return m_attributes_->check(key, v); }

// std::any EntryInMemory::get_attribute(std::string const& key) const { return m_attributes_->get(key); }

// std::any EntryInMemory::get_attribute(std::string const& key, std::any const& default_value) { return m_attributes_->get(key, default_value); }

// void EntryInMemory::set_attribute(std::string const& key, std::any const& v) { m_attributes_->set(key, v); }

// void EntryInMemory::remove_attribute(std::string const& key) { m_attributes_->erase(key); }

// Range<Iterator<std::pair<std::string, std::any>>> EntryInMemory::attributes() const { return std::move(m_attributes_->items()); }

// void EntryInMemory::clear_attributes() { m_attributes_->clear(); }

// //------------------------------------------------------------------------------------
// // node
// Entry* EntryInMemory::create() { return new EntryInMemory(); }

// Entry* EntryInMemory::copy() const { return new EntryInMemory(*this); }

// void EntryInMemory::resolve() {}

// Entry* EntryInMemory::create_child()
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// void EntryInMemory::as_scalar()
// {
//     if (m_content_->type_info() != typeid(ContentScalar))
//     {
//         m_content_.reset(new ContentScalar{});
//     }
// }

// void EntryInMemory::as_block()
// {
//     if (m_content_->type_info() != typeid(ContentBlock))
//     {
//         m_content_.reset(new ContentBlock{});
//     }
// }

// // as leaf node
// std::any EntryInMemory::get_scalar() const { return std::any{}; } // get value , if value is invalid then throw exception

// void EntryInMemory::set_scalar(std::any const& v) {}

// std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>
// EntryInMemory::get_raw_block() const
// {
//     return std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>>{nullptr, typeid(nullptr_t), {}};
// } // get block

// void EntryInMemory::set_raw_block(const std::shared_ptr<void>& data /*data pointer*/,
//                                   const std::type_info& t /*element type*/,
//                                   const std::vector<size_t>& dims /*dimensions*/)
// {
// }

// // as tree node

// size_t EntryInMemory::size() const
// {

//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         return dynamic_cast<const ContentArray*>(m_content_.get())->content.size();
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         return dynamic_cast<const ContentTable*>(m_content_.get())->content.size();
//     }
//     else
//     {
//         return 0;
//     }
// }

// Entry::range EntryInMemory::children()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<ContentArray*>(m_content_.get())->content;

//         return Entry::range(content.begin(), content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<ContentTable*>(m_content_.get())->content;

//         return Entry::range(content.begin(), content.end());
//     }
//     else
//     {
//         return Entry::range{};
//     }
// }

// Entry::const_range EntryInMemory::children() const
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Entry::const_range(content.begin(), content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Entry::const_range(content.begin(), content.end());
//     }
//     else
//     {
//         return Entry::range{};
//     }
// }

// void EntryInMemory::clear_children()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         dynamic_cast<ContentArray*>(m_content_.get())->content.clear();
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         dynamic_cast<ContentTable*>(m_content_.get())->content.clear();
//     }
// }

// void EntryInMemory::remove_child(Entry::iterator const&) {}

// void EntryInMemory::remove_children(Entry::range const&) {}

// Entry::iterator EntryInMemory::begin()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Entry::iterator(content.begin());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Entry::iterator(content.begin());
//     }
//     else
//     {
//         return Entry::iterator();
//     }
// }

// Entry::iterator EntryInMemory::end()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Entry::iterator(content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Entry::iterator(content.end());
//     }
//     else
//     {
//         return Entry::iterator();
//     }
// }

// Entry::const_iterator EntryInMemory::cbegin() const
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Entry::const_iterator(content.begin());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Entry::const_iterator(content.begin());
//     }
//     else
//     {
//         return Entry::const_iterator();
//     }
// }

// Entry::const_iterator EntryInMemory::cend() const
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Entry::const_iterator(content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Entry::const_iterator(content.end());
//     }
//     else
//     {
//         return Entry::const_iterator();
//     }
// }

Entry*
create_entry(const std::string& str)
{
    return new EntryImplement<entry_in_memory>();
}
} // namespace sp