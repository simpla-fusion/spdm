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

        virtual std::shared_ptr<Node> append(std::shared_ptr<Node> const &) = 0; // append node to tail of list , return reference of new node

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

    template <typename TAG, typename... OTHERS>
    class EntryTmpl : public Entry
    {
    public:
        typedef EntryTmpl<TAG, OTHERS...> this_type;

        EntryTmpl();

        EntryTmpl(this_type const &other);

        EntryTmpl(this_type &&other);

        ~EntryTmpl();

        EntryTmpl &operator=(this_type const &other);

        void swap(this_type &other);

        Entry *copy() const;

        Entry *move();

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        int type() const;

        //----------------------------------------------------------------------------------------------------------
        // attribute
        //----------------------------------------------------------------------------------------------------------
        bool has_attribute(std::string const &k) const; // if key exists then return true else return false

        bool check_attribute(std::string const &k, std::any const &v) const; // if key exists and value ==v then return true else return false

        std::any get_attribute(std::string const &key) const; // get attribute at key, if key does not exist return nullptr

        std::any get_attribute(std::string const &key, std::any const &default_value = std::any{}); // get attribute at key, if key does not exist return nullptr

        void set_attribute(std::string const &key, std::any const &v); // set attribute at key as v

        void remove_attribute(std::string const &key = ""); // remove attribute at key, if key=="" then remove all

        Range<Iterator<const std::pair<const std::string, std::any>>> attributes() const; // return reference of  all attributes

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------

        std::any get_scalar() const; // get value , if value is invalid then throw exception

        void set_scalar(std::any const &); // set value , if fail then throw exception

        std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> get_raw_block() const; // get block

        void set_raw_block(std::shared_ptr<char> const &, std::type_info const &, std::vector<size_t> const &); // set block

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        // function level 0
        std::shared_ptr<const Node> find_child(std::string const &) const; // return reference of child node , if key does not exists then insert new

        std::shared_ptr<Node> find_child(std::string const &); // return reference of child node , if key does not exists then insert new

        std::shared_ptr<Node> append(); // append node to tail of list , return reference of new node

        std::shared_ptr<Node> append(std::shared_ptr<Node> const &);

        void append(const Iterator<std::shared_ptr<Node>> &b, const Iterator<std::shared_ptr<Node>> &); // insert node to object

        std::shared_ptr<Node> insert(std::string const &key, std::shared_ptr<Node> const &n = nullptr); // insert node to object

        void insert(Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &b,
                    Iterator<std::pair<const std::string, std::shared_ptr<Node>>> const &e); // insert node to object

        std::shared_ptr<Node> child(std::string const &key); // get child, create new if key does not  exist

        // std::shared_ptr<const Node> child(std::string const &key) const; // get child, create new if key does not  exist

        std::shared_ptr<Node> child(int idx); // return reference i-th child node , if idx does not exists then throw exception

        // std::shared_ptr<const Node> child(int idx) const; // return reference i-th child node , if idx does not exists then throw exception

        void remove_child(std::string const &key); // remove child at key

        void remove_child(int idx); // remove i-th child

        void remove_children(); // remove children , set node.type => Null

        // std::pair<Iterator<const Node>, Iterator<const Node>> children() const; // reutrn list of children

        std::pair<Iterator<Node>, Iterator<Node>> children(); // reutrn list of children

        // level 1
        // std::pair<Iterator<const Node>, Iterator<const Node>> select(XPath const &path) const; // select from children

        std::pair<Iterator<Node>, Iterator<Node>> select(XPath const &path); // select from children

        // std::shared_ptr<const Node> select_one(XPath const &path) const; // return the first selected child

        std::shared_ptr<Node> select_one(XPath const &path); // return the first selected child

    private:
        std::unique_ptr<TAG> m_pimpl_;
    };

} // namespace sp
#endif //SP_ENTRY_H_