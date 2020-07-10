#ifndef SP_NODE_H_
#define SP_NODE_H_

#include "Range.h"
#include <algorithm>
#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
namespace sp
{

    enum NodeTag
    {
        Null = 0b0000, // value is invalid
        Scalar = 0b0010,
        Block = 0b0110,
        List = 0b0011,
        Object = 0b0111
    };
    class XPath;
    class Entry;
    class Node;

    class Node : public std::enable_shared_from_this<Node>
    {
    public:
        typedef Node this_type;
        typedef Iterator<Node *> iterator;
        typedef Range<iterator> range;

        Node(Node *parent, Entry *e);
        Node(Node *parent = nullptr, std::string const &backend = "");
        Node(this_type const &other);
        Node(this_type &&other);
        ~Node();
        this_type &operator=(this_type const &other);
        void swap(this_type &other);

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        int type() const;     // Null,Scalar,Block,List ,Object
        bool empty() const;   // if value is not valid then true else false
        bool is_root() const; // if  parent is null then true else false
        bool is_leaf() const; // if type in [Null,Scalar,Block]

        //----------------------------------------------------------------------------------------------------------
        // attribute
        //----------------------------------------------------------------------------------------------------------
        bool has_attribute(std::string const &k) const; // if key exists then return true else return false

        bool check_attribute(std::string const &k, std::any const &v) const; // if key exists and value ==v then return true else return false

        std::any get_attribute(std::string const &key) const; // get attribute at key, if key does not exist return nullptr

        template <typename U>
        U get_attribute(std::string const &key) const { return std::any_cast<U>(get_attribute(key)); }

        void set_attribute(std::string const &key, std::any const &v); // set attribute at key as v

        template <typename U, typename V>
        void set_attribute(std::string const &key, V const &v) { set_attribute(key, std::make_any<V>(v)); }

        void remove_attribute(std::string const &key = ""); // remove attribute at key, if key=="" then remove all

        Range<Iterator<const std::pair<const std::string, std::any>>> attributes() const; // return reference of  all attributes

        //----------------------------------------------------------------------------------------------------------
        // as leaf node,  need node.type = Scalar || Block
        //----------------------------------------------------------------------------------------------------------

        std::any get_scalar() const; // get value , if value is invalid then throw exception

        void set_scalar(std::any const &);

        std::tuple<std::shared_ptr<char>, std::type_info const &, std::vector<size_t>> get_raw_block() const; // get block

        void set_raw_block(std::shared_ptr<char> const & /*data pointer*/, std::type_info const & /*element type*/, std::vector<size_t> const & /*dimensions*/); // set block

        template <typename U, typename V>
        void set_value(V const &v) { set_scalar(std::make_any<U>(v)); }

        template <typename U>
        U get_value() const { return std::any_cast<U>(get_scalar()); }

        template <typename V, typename... Args>
        void set_block(std::shared_ptr<V> const &d, Args... args) { set_raw_block(std::reinterpret_pointer_cast<char>(d), typeid(V), std::vector<size_t>{std::forward<Args>(args)...}); }

        template <typename V, typename... Args>
        std::tuple<std::shared_ptr<V>, std::type_info const &, std::vector<size_t>> const get_block() const
        {
            auto blk = get_raw_block();
            return std::make_tuple(std::reinterpret_pointer_cast<char>(std::get<0>(blk)), std::get<1>(blk), std::get<2>(blk));
        }

        //----------------------------------------------------------------------------------------------------------
        // as tree node,  need node.type = List || Object
        //----------------------------------------------------------------------------------------------------------
        // function level 0
        Node &parent() const;                         // return parent node
        Node &child(std::string const &);             // return reference of child node , if key does not exists then insert new
        const Node &child(std::string const &) const; // return reference of child node , if key does not exists then insert new
        Node &child(int idx);                         // return reference i-th child node , if idx does not exists then throw exception
        const Node &child(int idx) const;             // return reference i-th child node , if idx does not exists then throw exception
        Node &append();                               // append node to tail of list , return reference of new node
        void remove_child(int idx);                   // remove i-th child
        void remove_child(std::string const &key);    // remove child at key
        void remove_children();                       // remove children , set node.type => Null

        Range<Iterator<std::string>> keys() const; // reutrn keys of children , only valid for Object Node
        range children() const;                    // reutrn list of children

        //----------------------------------------------------------------------------------------------------------
        // function level 1
        range select(XPath const &path) const;     // select from children
        Node &select_one(XPath const &path);       // return refernce of the first selected child  , if fail then throw exception
        Node &select_one(XPath const &path) const; // return refernce of the first selected child , if fail then throw exception
        Node &operator[](std::string const &path); // => select_one(XPath(path))
        Node &operator[](size_t);                  // => child(idx)

        //----------------------------------------------------------------------------------------------------------
        // function level 2
        ptrdiff_t distance(const this_type &target) const; // lenght of short path to target
        size_t depth() const;                              // parent.depth +1
        size_t height() const;                             // max(leaf.height) +1
        iterator first_child() const;                      // return iterator of the first child;
        range slibings() const;                            // return slibings
        range ancestor() const;                            // return ancestor
        range descendants() const;                         // return descendants
        range leaves() const;                              // return leave nodes in traversal order
        range path(this_type const &target) const;         // return the shortest path to target

        //----------------------------------------------------------------------------------------------------------
        // backend
        const Entry &entry() const; // return bundle of attributes
        Entry &entry();             // return bundle of attributes

    private:
        Node *m_parent_;
        std::unique_ptr<Entry> m_entry_;
    };

} // namespace sp
std::ostream &operator<<(std::ostream &os, sp::Node const &d);

#endif // SP_NODE_H_