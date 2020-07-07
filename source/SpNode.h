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

    struct SpEntry
    {
    public:
        class Attributes
        {
        public:
            Attributes *copy() const { return new Attributes(*this); }

            std::any get(std::string const &name) const;         // get attribute, return nullptr is name does not exist
            int set(std::string const &name, std::any const &v); // set attribute
            int remove(std::string const &name);                 // remove attribuet
        };

        class Content
        {
        public:
            Content *copy() const { return new Content(*this); }
        };

        SpEntry() : m_content_(nullptr), m_attributes_(nullptr){};
        ~SpEntry() = default;
        SpEntry(SpEntry const &other) : m_content_(other.m_content_.copy()), m_attributes_(other.m_attributes_->copy()) {}
        SpEntry(SpEntry &&other) : m_content_(std::move(other.m_content_)), m_attributes_(std::move(other.m_attributes_)) {}

        SpEntry &operator=(SpEntry const &other)
        {
            SpEntry(other).swap(*this);
            return *this;
        }
        SpEntry &swap(SpEntry &other)
        {
            std::swap(m_content_, other.m_content_);
            std::swap(m_attributes_, other.m_attributes_);
        }
        SpEntry *copy() const { return new SpEntry(*this); };

        SpNode::TypeOfNode type() const { return SpNode::TypeOfNode::Null; };

        // attributes
        const auto &attributes() const { return *m_attributes_; } // return list of attributes
        auto &attributes() { return *m_attributes_; }             // return list of attributes

        //----------------------------------------------------------------------------------------------------------
        // content
        auto &content() { return *m_content_; }
        const auto &content() const { return *m_content_; }

    private:
        std::unique_ptr<Content> m_content_;
        std::unique_ptr<Attributes> m_attributes_;
    };
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

        SpNode() : m_parent_(nullptr), m_entry_(nullptr){};
        ~SpNode() = default;
        SpNode(SpNode const &other) : m_parent_(other.m_parent_), m_entry_(other.m_entry_->copy()) {}
        SpNode(SpNode &&other) : m_parent_(other.m_parent_), m_entry_(std::move(other.m_entry_)) { other.m_entry_.reset(); }
        SpNode(std::shared_ptr<SpNode> const &parent) : SpNode(), m_parent_(parent) {}

        SpNode &operator=(SpNode const &other)
        {
            SpNode(other).swap(*this);
            return *this;
        }

        SpNode &swap(SpNode &other)
        {
            std::swap(m_parent_, other.m_parent_);
            std::swap(m_entry_, other.m_entry_);
        }

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        bool is_root() const { return m_parent_ == nullptr; }

        bool is_leaf() const { return m_entry_ == nullptr || m_entry_->is_leaf(); }

        size_t depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

        bool same_as(this_type const &other) const;

        // attributes
        const auto &attributes() const { return m_entry_->attributes(); } // return list of attributes
        auto &attributes() { return m_entry_->attributes(); }             // return list of attributes

        //----------------------------------------------------------------------------------------------------------
        // content
        auto &content() { return m_entry_->content(); }
        const auto &content() const { return m_entry_->content(); }

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        std::shared_ptr<SpNode> operator[](T const &v) { return child(v); }

        //----------------------------------------------------------------------------------------------------------
        // level 0
        std::shared_ptr<SpNode> parent() const;            // return parent node
        void remove();                                     // remove self from parent
        int remove_child(int idx);                         // remove child with key, return true if key exists
        int remove_child(std::string const &key);          // remove child with key, return true if key exists
        int remove_child(std::shared_ptr<SpNode> const &); // remove child with key, return true if key exists
        int remove_child(iterator);                        // remove child with key, return true if key exists

        // as object
        std::shared_ptr<SpNode> child(std::string const &) const; // return node at key,  if key does not exist then throw exception
        std::shared_ptr<SpNode> child(std::string const &);       // return node at key,  if key does not exist then create one

        // as list
        std::shared_ptr<SpNode> child(int);                                         // return node at idx,  if idx >size() then throw exception
        std::shared_ptr<SpNode> child(int) const;                                   // return node at idx,  if idx >size() then throw exception
        std::shared_ptr<SpNode> append(std::shared_ptr<SpNode> const &n = nullptr); // insert new child node after last child

        //traversal

        iterator first_child() const; // return first child node, next(first_child()) return the nearest slibing
        range children() const;       // return children

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
        std::shared_ptr<SpNode> m_parent_;
        std::unique_ptr<SpEntry> m_entry_;
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