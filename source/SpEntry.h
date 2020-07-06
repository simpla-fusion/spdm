#ifndef SPDB_IMPLEMENT_H_
#define SPDB_IMPLEMENT_H_
#include "SpNode.h"
#include <any>
#include <map>
#include <memory>
namespace sp
{

    template <>
    class SpEntry::ContentT<SpNode::TypeOfNode::Scalar> : public SpEntry::Content
    {
    public:
    };
    // class SpEntry : public std::enable_shared_from_this<SpEntry>
    // {
    // public:
    //     friend class SpNode;
    //     typedef std::shared_ptr<SpNode> node_type;
    //     typedef std::pair<node_type, node_type> range_type;

    //     SpEntry(){};
    //     SpEntry(SpEntry const &) {}
    //     SpEntry(SpEntry &&) {}
    //     SpEntry(std::shared_ptr<SpNode> const &parent) : m_parent_(parent) {}
    //     virtual ~SpEntry(){};
    //     void swap(SpEntry &other) { std::swap(m_parent_, other.m_parent_); }

    //     virtual SpNode::TypeOfNode type() const { return SpNode::TypeOfNode::Null; }; //

    //     std::shared_ptr<SpNode> parent() const { return m_parent_; }

    //     size_t depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }; // distance(root())

    //     virtual size_t size() const = 0;

    //     virtual void remove() = 0;                                             // remove self
    //     virtual std::shared_ptr<SpEntry> copy() const = 0;                     //
    //     virtual std::shared_ptr<SpEntry> create(SpNode::TypeOfNode) const = 0; //

    //     virtual std::map<std::string, std::any> attributes() const = 0;        // return list of attributes
    //     virtual std::any attribute(std::string const &name) const = 0;         // get attribute, return nullptr is name does not exist
    //     virtual int attribute(std::string const &name, std::any const &v) = 0; // set attribute
    //     virtual int remove_attribute(std::string const &name) = 0;             // remove attribute

    //     virtual void value(std::any const &) = 0;  // set value
    //     virtual std::any &value() = 0;             // get value
    //     virtual std::any const &value() const = 0; // get value

    //     virtual range_type children() const = 0;      // return children
    //     virtual node_type first_child() const = 0;    // return first child node
    //     virtual node_type child(int) = 0;             // return node at idx,  if idx >size() then throw exception
    //     virtual node_type child(int) const = 0;       // return node at idx,  if idx >size() then throw exception
    //     virtual node_type insert_before(int pos) = 0; // insert new child node before pos, return new node
    //     virtual node_type insert_after(int pos) = 0;  // insert new child node after pos, return new node
    //     virtual node_type prepend() = 0;              // insert new child node before first child
    //     virtual node_type append() = 0;               // insert new child node afater last child

    //     virtual node_type child(std::string const &) const = 0; // return node at key,  if key does not exist then throw exception
    //     virtual node_type child(std::string const &) = 0;       // return node at key,  if key does not exist then create one

    //     virtual int remove_child(std::string const &key) = 0; // remove child with key, return true if key exists
    //     virtual int remove_child(int idx) = 0;                // remove child at pos, return true if idx exists

    //     virtual range_type select(SpXPath const &path) const = 0;    // select from children
    //     virtual node_type select_one(SpXPath const &path) const = 0; // return the first selected child

    // protected:
    //     std::shared_ptr<SpNode> m_parent_;
    // };

} // namespace sp
#endif //SPDB_IMPLEMENT_H_