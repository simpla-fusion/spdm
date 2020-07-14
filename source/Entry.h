#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Node.h"

namespace sp
{
class EntryInterface
{
public:
    EntryInterface() = default;

    virtual ~EntryInterface() = default;

    virtual TypeTag type_tag() const = 0;

    virtual EntryInterface* create() = 0;

    virtual EntryInterface* copy() const = 0;

    virtual Node* create_child() = 0;

    virtual void resolve() = 0;

    virtual void as_scalar() = 0;

    virtual void as_block() = 0;

    virtual void as_array() = 0;

    virtual void as_table() = 0;

    // attributes
    virtual bool has_attribute(std::string const& key) const = 0;

    virtual bool check_attribute(std::string const& key, std::any const& v) const = 0;

    virtual void set_attribute(const std::string&, const std::any&) = 0;

    virtual std::any get_attribute(const std::string&) const = 0;

    virtual std::any get_attribute(std::string const& key, std::any const& default_value) = 0;

    virtual void remove_attribute(const std::string&) = 0;

    virtual Range<Iterator<std::pair<std::string, std::any>>> attributes() const = 0;

    virtual void clear_attributes() = 0;

    // as leaf node
    virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

    virtual void set_scalar(std::any const&) = 0;

    virtual std::tuple<std::shared_ptr<void>, const std::type_info&, std::vector<size_t>> get_raw_block() const = 0; // get block

    virtual void set_raw_block(const std::shared_ptr<void>& /*data pointer*/,
                               const std::type_info& /*element type*/,
                               const std::vector<size_t>& /*dimensions*/) = 0; // set block

    // as tree node

    virtual size_t size() const = 0;

    virtual Node::range children() = 0; // reutrn list of children

    virtual Node::const_range children() const = 0; // reutrn list of children

    virtual void clear_children() = 0;

    virtual void remove_child(Node::iterator const&) = 0;

    virtual void remove_children(Node::range const&) = 0;

    virtual Node::iterator begin() = 0;

    virtual Node::iterator end() = 0;

    virtual Node::const_iterator cbegin() const = 0;

    virtual Node::const_iterator cend() const = 0;

    // as array

    virtual std::shared_ptr<Node> push_back(const std::shared_ptr<Node>& p = nullptr) = 0;

    virtual std::shared_ptr<Node> push_back(Node&&) = 0;

    virtual std::shared_ptr<Node> push_back(const Node&) = 0;

    virtual Node::range push_back(const Node::iterator& b, const Node::iterator& e) = 0;

    virtual std::shared_ptr<Node> at(int idx) = 0;

    virtual std::shared_ptr<const Node> at(int idx) const = 0;

    virtual std::shared_ptr<Node> find_child(size_t) = 0;

    virtual std::shared_ptr<const Node> find_child(size_t) const = 0;

    // as table

    virtual Node::const_range_kv items() const = 0;

    virtual Node::range_kv items() = 0;

    virtual std::shared_ptr<Node> insert(const std::string& k, std::shared_ptr<Node> const& node) = 0;

    virtual Node::range_kv insert(const Node::iterator_kv& b, const Node::iterator_kv& e) = 0;

    virtual std::shared_ptr<Node> at(const std::string& key) = 0;

    virtual std::shared_ptr<const Node> at(const std::string& idx) const = 0;

    virtual std::shared_ptr<Node> find_child(const std::string&) = 0;

    virtual std::shared_ptr<const Node> find_child(const std::string&) const = 0;
};

EntryInterface* create_entry(const std::string& backend = "");

} // namespace sp

#endif // SP_ENTRY_H_
