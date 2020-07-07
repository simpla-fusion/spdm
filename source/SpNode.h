#ifndef SP_NODE_H_
#define SP_NODE_H_
#include "SpUtil.h"
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
    class SpXPath;
    class SpNode;
    class SpDocument;

    class SpXPath
    {
    public:
        SpXPath(std::string const &path = "");
        SpXPath(const char *path);
        ~SpXPath() = default;

        SpXPath(SpXPath &&) = default;
        SpXPath(SpXPath const &) = default;
        SpXPath &operator=(SpXPath const &) = default;

        const std::string &str() const;

        SpXPath operator/(std::string const &suffix) const;
        operator std::string() const;

    private:
        std::string m_path_;
    };

    struct SpEntry;

    class SpNode : public std::enable_shared_from_this<SpNode>
    {
    public:
        enum TypeOfNode
        {
            Null,
            Scalar,
            Block,
            List,
            Object
        };

        class iterator;
        class range;

        typedef SpNode this_type;

        SpNode();
        virtual ~SpNode();
        SpNode(SpNode const &parent);
        SpNode(SpNode &&other);
        SpNode(std::shared_ptr<SpEntry> const &entry);

        SpNode &operator=(SpNode const &);

        SpNode &swap(SpNode &other);

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        TypeOfNode type() const;
        bool is_root() const; // parent().empty() is true
        bool is_leaf() const; // children().size() =0
        size_t depth() const; // distance(root())
        bool empty() const;

        bool same_as(this_type const &other) const;

        bool is_null() const { return type() == TypeOfNode::Null; }
        bool is_scalar() const { return type() == TypeOfNode::Scalar; }
        bool is_block() const { return type() == TypeOfNode::Block; }
        bool is_list() const { return type() == TypeOfNode::List; }
        bool is_object() const { return type() == TypeOfNode::Object; }

        //----------------------------------------------------------------------------------------------------------
        // attribute
        std::map<std::string, std::any> attributes() const;        // return list of attributes
        std::any attribute(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int attribute(std::string const &name, std::any const &v); // set attribute
        int remove_attribute(std::string const &name);             // remove attribuet

        //----------------------------------------------------------------------------------------------------------
        // value
        void value(std::any);
        std::any value() const;

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        SpNode operator[](T const &v) { return child(v); }

        //----------------------------------------------------------------------------------------------------------
        // level 0
        void remove(); // remove self

        SpNode parent() const;      // return parent node
        SpNode first_child() const; // return first child node
        range children() const;     // return children

        // as object
        SpNode child(std::string const &) const;  // return node at key,  if key does not exist then throw exception
        SpNode child(std::string const &);        // return node at key,  if key does not exist then create one
        int remove_child(std::string const &key); // remove child with key, return true if key exists

        // as list
        SpNode child(int);         // return node at idx,  if idx >size() then throw exception
        SpNode child(int) const;   // return node at idx,  if idx >size() then throw exception
        SpNode append();           // insert new child node afater last child
        int remove_child(int idx); // remove child at pos, return true if idx exists

        //----------------------------------------------------------------------------------------------------------
        // level 1
        range select(SpXPath const &path) const;      // select from children
        SpNode select_one(SpXPath const &path) const; // return the first selected child

        //----------------------------------------------------------------------------------------------------------
        // level 2
        range ancestor() const;                            // return ancestor
        range descendants() const;                         // return descendants
        range leaves() const;                              // return leave nodes in traversal order
        range slibings() const;                            // return slibings
        range path(SpNode const &target) const;            // return the shortest path to target
        ptrdiff_t distance(const this_type &target) const; // lenght of short path to target

    private:
        std::shared_ptr<SpEntry> m_entry_;
    };

    std::ostream &operator<<(std::ostream &os, SpNode const &d) { return d.repr(os); }
    //----------------------------------------------------------------------------------------------------------

    class SpNode::iterator : public std::iterator<std::input_iterator_tag, SpNode>
    {
    public:
        iterator();
        virtual ~iterator();
        iterator(iterator const &);
        iterator(iterator &&);
        iterator &swap(iterator &other);
        iterator &operator=(iterator const &other) { return iterator(other).swap(*this); }
       
        iterator(std::shared_ptr<SpNode> const &);

        bool operator==(iterator const &other) const { return equal(other); }
        bool operator!=(iterator const &other) const { return !equal(other); }
        ptrdiff_t operator-(iterator const &other) const { return distance(other); }

        iterator &operator++()
        {
            next().swap(*this);
            return *this;
        }
        iterator operator++(int)
        {
            iterator res(*this);
            next().swap(*this);
            return res;
        }
        reference operator*() { return *self(); }
        pointer operator->() { return self(); }

        iterator next() const;
        bool equal(iterator const &) const;
        ptrdiff_t distance(iterator const &) const;
        pointer self();

    private:
        struct pimpl_s;
        std::unique_ptr<pimpl_s> m_pimpl_;
    };

    class SpNode::range : public std::pair<SpNode::iterator, SpNode::iterator>
    {

    public:
        typedef std::pair<iterator, iterator> base_type;

        using base_type::first;
        using base_type::second;

        template <typename... Args>
        range(Args &&... args) : base_type(SpNode::iterator(std::forward<Args>(args))...) {}
        // virtual ~range(){};

        ptrdiff_t size() { return std::distance(first, second); };

        iterator begin() const { return first; };
        iterator end() const { return second; }
    };

} // namespace sp

#endif // SP_NODE_H_