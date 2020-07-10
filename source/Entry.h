#ifndef SP_ENTRY_H_
#define SP_ENTRY_H_
#include "Range.h"
#include "Util.h"
#include "XPath.h"
#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
namespace sp
{
    class Node;

    class Entry
    {
    public:
        typedef Entry this_type;
        static const char SP_DEFAULT_KEY_OF_CONTENT = '_';

    protected:
        Node *m_self_;

    public:
        Entry();
        Entry(Entry const &other);
        Entry(Entry &&other);
        virtual ~Entry();

        void bind(Node *self) { m_self_ = self; }

        Node &node() const { return *m_self_; }

        void swap(Entry &other);

        static Entry *create(std::string const &backend = "");

        virtual Entry *copy() const = 0;

        virtual Entry *move() = 0;

        virtual std::ostream &repr(std::ostream &os) const = 0; // represent object as string and push ostream

        virtual int type() const = 0;

        //----------------------------------------------------------------------------------------------------------
        // attribute
        //----------------------------------------------------------------------------------------------------------
        virtual bool has_attribute(std::string const &k) const = 0; // if key exists then return true else return false

        virtual bool check_attribute(std::string const &k, std::any const &v) const = 0; // if key exists and value ==v then return true else return false

        virtual std::any get_attribute(std::string const &key) const = 0; // get attribute at key, if key does not exist return nullptr

        virtual std::any get_attribute(std::string const &key, std::any const &default_value = std::any{}) = 0; // get attribute at key, if key does not exist return nullptr

        virtual void set_attribute(std::string const &key, std::any const &v) = 0; // set attribute at key as v

        virtual void remove_attribute(std::string const &key = "") = 0; // remove attribute at key, if key=="" then remove all

        virtual Range<Iterator<const std::pair<const std::string, std::any>>> attributes() const = 0; // return reference of  all attributes

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------

        virtual std::any get_scalar() const = 0; // get value , if value is invalid then throw exception

        virtual void set_scalar(std::any const &) = 0; // set value , if fail then throw exception

        virtual std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> get_raw_block() const = 0; // get block

        virtual void set_raw_block(std::shared_ptr<char> const &, std::type_info const &, std::vector<size_t> const &) = 0; // set block

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        // function level 0
        virtual std::shared_ptr<const Node> find_child(std::string const &) const = 0; // return reference of child node , if key does not exists then insert new

        virtual std::shared_ptr<Node> find_child(std::string const &) = 0; // return reference of child node , if key does not exists then insert new

        virtual std::shared_ptr<Node> append() = 0; // append node to tail of list , return reference of new node

        virtual std::shared_ptr<Node> append(std::shared_ptr<Node>  const &) = 0; // append node to tail of list , return reference of new node

        virtual void append(const Iterator<std::shared_ptr<Node>> &b, const Iterator<std::shared_ptr<Node>> &) = 0; // insert node to object

        virtual std::shared_ptr<Node> insert(std::string const &key, std::shared_ptr<Node> const &n = nullptr) = 0; // insert node to object

        virtual void insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &b,
                            Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &e) = 0; // insert node to object

        virtual std::shared_ptr<Node> child(std::string const &key) = 0; // get child, create new if key does not  exist

        // virtual std::shared_ptr<const Node> child(std::string const &key) const = 0; // get child, create new if key does not  exist

        virtual std::shared_ptr<Node> child(int idx) = 0; // return reference i-th child node , if idx does not exists then throw exception

        // virtual std::shared_ptr<const Node> child(int idx) const = 0; // return reference i-th child node , if idx does not exists then throw exception

        virtual void remove_child(std::string const &key) = 0; // remove child at key

        virtual void remove_child(int idx) = 0; // remove i-th child

        virtual void remove_children() = 0; // remove children , set node.type => Null

        // virtual std::pair<Iterator<const Node>, Iterator<const Node>> children() const = 0; // reutrn list of children

        virtual std::pair<Iterator<Node>, Iterator<Node>> children() = 0; // reutrn list of children

        // level 1
        // virtual std::pair<Iterator<const Node>, Iterator<const Node>> select(XPath const &path) const = 0; // select from children

        virtual std::pair<Iterator<Node>, Iterator<Node>> select(XPath const &path) = 0; // select from children

        // virtual std::shared_ptr<const Node> select_one(XPath const &path) const = 0; // return the first selected child

        virtual std::shared_ptr<Node> select_one(XPath const &path) = 0; // return the first selected child
    };

} // namespace sp
#endif //SP_ENTRY_H_