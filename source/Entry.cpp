#include "Entry.h"
#include "Node.h"
#include "utility/Logger.h"
namespace sp
{

EntryInterface<TypeTag::Null>* EntryInterface<TypeTag::Null>::as_interface(TypeTag tag)
{
    if (tag == type_tag())
    {
        return this;
    }
    else
    {
        NOT_IMPLEMENTED;
        return nullptr;
    }
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
//     std::vector<std::shared_ptr<Node>> content;
//     ContentArray() {}
//     ContentArray(ContentArray const& other) : content(other.content) {}
//     ContentArray(ContentArray&& other) : content(std::move(other.content)) {}
//     std::type_info const& type_info() const { return typeid(ContentArray); }
//     Content* copy() const { return new ContentArray{*this}; };
//     Content* move() { return new ContentArray{std::move(*this)}; };
// };

// struct ContentTable : public Content
// {
//     std::map<std::string, std::shared_ptr<Node>> content;
//     ContentTable() {}
//     ContentTable(ContentTable const& other) : content(other.content) {}
//     ContentTable(ContentTable&& other) : content(std::move(other.content)) {}
//     std::type_info const& type_info() const { return typeid(ContentTable); }
//     Content* copy() const { return new ContentTable{*this}; };
//     Content* move() { return new ContentTable{std::move(*this)}; };
// };

// class EntryInMemory : public EntryInterface
// {
// public:
//     EntryInMemory();

//     EntryInMemory(EntryInMemory const&);

//     EntryInMemory(EntryInMemory&&);

//     ~EntryInMemory() override;

//     // attributes
//     bool has_attribute(std::string const& key) const override;

//     bool check_attribute(std::string const& key, std::any const& v) const override;

//     void set_attribute(const std::string&, const std::any&) override;

//     std::any get_attribute(const std::string&) const override;

//     std::any get_attribute(std::string const& key, std::any const& default_value) override;

//     void remove_attribute(const std::string&) override;

//     Range<Iterator<std::pair<std::string, std::any>>> attributes() const override;

//     void clear_attributes();

//     // node
//     TypeTag type_tag() const override;

//     EntryInterface* create() override;

//     EntryInterface* copy() const override;

//     void resolve();

//     Node* create_child();

//     void as_scalar();

//     void as_block();

//     void as_array();

//     void as_table();

//     ContentScalar& as_scalar_();

//     ContentBlock& as_block_();

//     ContentArray& as_array_();

//     ContentTable& as_table_();

//     //------------------------------------------------------------------------------------------------
//     // as leaf node
//     std::any get_scalar() const override; // get value , if value is invalid then throw exception

//     void set_scalar(std::any const&) override;

//     std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const override; // get block

//     void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
//                        const std::type_info& /*element type*/,
//                        const std::vector<size_t>& /*dimensions*/) override; // set block
//     //------------------------------------------------------------------------------------------------
//     // as tree node

//     size_t size() const override;

//     Node::range children() override; // reutrn list of children

//     Node::const_range children() const override; // reutrn list of children

//     void clear_children() override;

//     void remove_child(Node::iterator const&) override;

//     void remove_children(Node::range const&) override;

//     Node::iterator begin() override;

//     Node::iterator end() override;

//     Node::const_iterator cbegin() const override;

//     Node::const_iterator cend() const override;

//     // as array

//     std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr) override;

//     std::shared_ptr<Node> push_back(Node&&) override;

//     std::shared_ptr<Node> push_back(const Node&) override;

//     Node::range push_back(const Node::iterator& b, const Node::iterator& e) override;

//     std::shared_ptr<Node> at(int idx) override;

//     std::shared_ptr<const Node> at(int idx) const override;

//     std::shared_ptr<Node> find_child(size_t) override;

//     std::shared_ptr<const Node> find_child(size_t) const override;

//     // as table

//     Node::const_range_kv items() const override;

//     Node::range_kv items() override;

//     std::shared_ptr<Node> insert(const std::string& k, const std::shared_ptr<Node>& node) override;

//     Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e) override;

//     std::shared_ptr<Node> at(const std::string& key) override;

//     std::shared_ptr<const Node> at(const std::string& idx) const override;

//     std::shared_ptr<Node> find_child(const std::string&) override;

//     std::shared_ptr<const Node> find_child(const std::string&) const override;

// private:
//     std::unique_ptr<Attributes> m_attributes_;
//     std::unique_ptr<Content> m_content_;
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
// EntryInterface* EntryInMemory::create() { return new EntryInMemory(); }

// EntryInterface* EntryInMemory::copy() const { return new EntryInMemory(*this); }

// void EntryInMemory::resolve() {}

// Node* EntryInMemory::create_child()
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

// void EntryInMemory::as_array() { NOT_IMPLEMENTED; }

// void EntryInMemory::as_table() { NOT_IMPLEMENTED; }

// TypeTag EntryInMemory::type_tag() const
// {
//     TypeTag res = TypeTag::Null;

//     if (m_content_->type_info() == typeid(ContentScalar))
//     {
//         res = TypeTag::Scalar;
//     }
//     else if (m_content_->type_info() == typeid(ContentBlock))
//     {
//         res = TypeTag::Block;
//     }
//     else if (m_content_->type_info() == typeid(ContentArray))
//     {
//         res = TypeTag::Array;
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         res = TypeTag::Table;
//     }
//     else
//     {
//         res = TypeTag::Null;
//     }

//     return res;
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

// Node::range EntryInMemory::children()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<ContentArray*>(m_content_.get())->content;

//         return Node::range(content.begin(), content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<ContentTable*>(m_content_.get())->content;

//         return Node::range(content.begin(), content.end());
//     }
//     else
//     {
//         return Node::range{};
//     }
// }

// Node::const_range EntryInMemory::children() const
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Node::const_range(content.begin(), content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Node::const_range(content.begin(), content.end());
//     }
//     else
//     {
//         return Node::range{};
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

// void EntryInMemory::remove_child(Node::iterator const&) {}

// void EntryInMemory::remove_children(Node::range const&) {}

// Node::iterator EntryInMemory::begin()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Node::iterator(content.begin());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Node::iterator(content.begin());
//     }
//     else
//     {
//         return Node::iterator();
//     }
// }

// Node::iterator EntryInMemory::end()
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Node::iterator(content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Node::iterator(content.end());
//     }
//     else
//     {
//         return Node::iterator();
//     }
// }

// Node::const_iterator EntryInMemory::cbegin() const
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Node::const_iterator(content.begin());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Node::const_iterator(content.begin());
//     }
//     else
//     {
//         return Node::const_iterator();
//     }
// }

// Node::const_iterator EntryInMemory::cend() const
// {
//     if (m_content_->type_info() == typeid(ContentArray))
//     {
//         const auto& content = dynamic_cast<const ContentArray*>(m_content_.get())->content;

//         return Node::const_iterator(content.end());
//     }
//     else if (m_content_->type_info() == typeid(ContentTable))
//     {
//         const auto& content = dynamic_cast<const ContentTable*>(m_content_.get())->content;

//         return Node::const_iterator(content.end());
//     }
//     else
//     {
//         return Node::const_iterator();
//     }
// }

// // as array

// std::shared_ptr<Node> EntryInMemory::push_back(const std::shared_ptr<Node>& p)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Node> EntryInMemory::push_back(Node&&)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Node> EntryInMemory::push_back(const Node&)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// Node::range EntryInMemory::push_back(const Node::iterator& b, const Node::iterator& e)
// {
//     NOT_IMPLEMENTED;
//     return Node::range{};
// }

// std::shared_ptr<Node> EntryInMemory::at(int idx)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<const Node> EntryInMemory::at(int idx) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Node> EntryInMemory::find_child(size_t)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<const Node> EntryInMemory::find_child(size_t) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// // as table

// Node::const_range_kv EntryInMemory::items() const
// {
//     NOT_IMPLEMENTED;
//     return Node::const_range_kv{};
// }

// Node::range_kv EntryInMemory::items()
// {
//     NOT_IMPLEMENTED;
//     return Node::range_kv{};
// }

// std::shared_ptr<Node> EntryInMemory::insert(const std::string& k, const std::shared_ptr<Node>& node)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// Node::range_kv EntryInMemory::insert(const Node::iterator_kv& b, const Node::iterator_kv& e)
// {
//     NOT_IMPLEMENTED;
//     return Node::range_kv{};
// }

// std::shared_ptr<Node> EntryInMemory::at(const std::string& key)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<const Node> EntryInMemory::at(const std::string& idx) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<Node> EntryInMemory::find_child(const std::string&)
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

// std::shared_ptr<const Node> EntryInMemory::find_child(const std::string&) const
// {
//     NOT_IMPLEMENTED;
//     return nullptr;
// }

struct entry_in_memory
{
    entry_in_memory* create() const { return new entry_in_memory(); }
    entry_in_memory* copy() const { return new entry_in_memory{*this}; }
    TypeTag type_tag() const { return m_type_tag_; }

    TypeTag m_type_tag_;
};
template <>
Node* Entry<entry_in_memory>::create_child()
{
    NOT_IMPLEMENTED;
    return nullptr;
}

template <>
void Entry<entry_in_memory>::resolve()
{
    NOT_IMPLEMENTED;
}

template <>
bool Entry<entry_in_memory>::has_attribute(std::string const& key) const
{
    NOT_IMPLEMENTED;
    return false;
}
template <>
bool Entry<entry_in_memory>::check_attribute(std::string const& key, std::any const& v) const
{
    NOT_IMPLEMENTED;
    return false;
}
template <>
void Entry<entry_in_memory>::set_attribute(const std::string&, const std::any&)
{
    NOT_IMPLEMENTED;
}
template <>
std::any Entry<entry_in_memory>::get_attribute(const std::string&) const
{
    NOT_IMPLEMENTED;
    return std::any{};
}
template <>
std::any Entry<entry_in_memory>::get_attribute(std::string const& key, std::any const& default_value)
{
    NOT_IMPLEMENTED;
    return std::any{};
}
template <>
void Entry<entry_in_memory>::remove_attribute(const std::string&)
{
    NOT_IMPLEMENTED;
}
template <>
Range<Iterator<std::pair<std::string, std::any>>>
Entry<entry_in_memory>::attributes() const
{
    NOT_IMPLEMENTED;
    return Range<Iterator<std::pair<std::string, std::any>>>{};
}
template <>
void Entry<entry_in_memory>::clear_attributes()
{
    NOT_IMPLEMENTED;
}

EntryInterface<>*
create_entry(const std::string& str)
{
    return new Entry<entry_in_memory>();
}
} // namespace sp