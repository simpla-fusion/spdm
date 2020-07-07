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

    enum NodeTag
    {
        Null,
        Scalar,
        Block,
        List,
        Object
    };
    class XPath;

    class Entry;
    class Attributes;
    class Content;

    class Node;

    //##############################################################################################################
    class XPath
    {
    public:
        XPath(std::string const &path = "");
        XPath(const char *path);
        ~XPath() = default;

        XPath(XPath &&) = default;
        XPath(XPath const &) = default;
        XPath &operator=(XPath const &) = default;

        const std::string &str() const;

        XPath operator/(std::string const &suffix) const;
        operator std::string() const;

    private:
        std::string m_path_;
    };

    //##############################################################################################################
    template <typename U>
    class Iterator : public std::iterator<std::input_iterator_tag, U>
    {
    public:
        typedef std::iterator<std::input_iterator_tag, U> base_type;

        typedef Iterator<U> iterator;

        using typename base_type::pointer;
        using typename base_type::reference;
        using typename base_type::value_type;

        typedef std::function<value_type(value_type const &)> next_function_type;

        Iterator() : m_self_(nullptr) {}
        Iterator(value_type first, next_function_type next_fun) : m_self_(first), m_next_(next_fun) { ; }
        Iterator(iterator const &other) : m_self_(other.m_self_), m_next_(other.m_next_) {}
        Iterator(iterator &&other) : m_self_(other.m_self_), m_next_(std::move(other.m_next_)) { other.m_self_ = nullptr; }

        ~Iterator() {}

        void swap(iterator &other)
        {
            std::swap(m_self_, other.m_self_);
            std::swap(m_next_, other.m_next_);
        }

        iterator &operator=(iterator const &other)
        {
            iterator(other).swap(*this);
            return *this;
        }

        bool operator==(iterator const &other) const { return m_self_ == other.m_self_; }
        bool operator!=(iterator const &other) const { return m_self_ != other.m_self_; }

        iterator &operator++()
        {
            m_self_ = m_next_(m_self_);

            return *this;
        }
        iterator operator++(int)
        {
            iterator res(*this);
            m_self_ = m_next_(m_self_);
            return res;
        }
        reference operator*() { return *m_self_; }
        pointer operator->() { return m_self_.get(); }

    private:
        value_type m_self_;
        next_function_type m_next_;
    };

    //##############################################################################################################
    template <typename T0, typename T1 = T0>
    class Range : public std::pair<T0, T1>
    {

    public:
        typedef std::pair<T0, T1> base_type;

        using base_type::first;
        using base_type::second;

        template <typename U0, typename U1>
        Range(U0 const &first, U1 const &second) : base_type(T0(first), T2(second)) {}

        template <typename U0, typename U1>
        Range(std::pair<U0, U1> const &p) : base_type(T0(p.first), T2(p.second)) {}

        // virtual ~range(){};

        ptrdiff_t size() { return std::distance(first, second); };

        T0 begin() const { return first; };
        T1 end() const { return second; }
    };

    //##############################################################################################################
    class Attributes
    {
    public:
        typedef Attributes this_type;

        static Attributes *create();
        Attributes *copy() const;

        Attributes() = default;
        Attributes(this_type const &) = default;
        Attributes(this_type &&) = default;
        virtual ~Attributes() = default;

        Attributes &operator=(this_type const &other);

        void swap(this_type &);

        std::any get(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int set(std::string const &name, std::any const &v); // set attribute
        int remove(std::string const &name);                 // remove attribuet
    };

    //##############################################################################################################
    class Content
    {
    public:
        typedef Content this_type;

        static Content *create(int tag);
        Content *as(int tag);
        Content *copy() const;

        Content() = default;
        virtual ~Content() = default;
        Content(this_type const &) = default;
        Content(this_type &&) = default;
        Content &operator=(this_type const &other);
        void swap(Content &);

        int type() const;
    };

    //##############################################################################################################
    class Entry : public std::enable_shared_from_this<Entry>
    {
    public:
        typedef Entry this_type;

        static Entry *create(int tag = 0);
        Entry *copy() const;

        Entry(int tag = 0);
        Entry(this_type const &other);
        Entry(this_type &&other);
        virtual ~Entry();
        this_type &operator=(this_type const &other);
        void swap(this_type &other);

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        int type() const;

        // attributes
        const Attributes &attributes() const { return *m_attributes_; } // return list of attributes
        Attributes &attributes() { return *m_attributes_; }             // return list of attributes

        //----------------------------------------------------------------------------------------------------------
        // content
        Content &content() { return *m_content_; }
        const Content &content() const { return *m_content_; }

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        std::shared_ptr<this_type> operator[](T const &v) { return child(v); }

        //----------------------------------------------------------------------------------------------------------
        // level 0
        std::shared_ptr<Node> parent() const; // return parent node
        void remove();                          // remove self from parent

        //----------------------------------------------------------------------------------------------------------
        // level 1
        // range select(XPath const &path) const;                       // select from children
        // std::shared_ptr<Node> select_one(XPath const &path) const; // return the first selected child

    protected:
        Entry(Attributes *attr, Content *content);

    private:
        std::unique_ptr<Content> m_content_;
        std::unique_ptr<Attributes> m_attributes_;
    };

    //##############################################################################################################
    class Node : public std::enable_shared_from_this<Node>
    {
    public:
        typedef Node this_type;
        typedef Iterator<std::shared_ptr<this_type>> iterator;
        typedef Range<iterator, iterator> range;

        Node(std::shared_ptr<Node> const &parent = nullptr, int tag = NodeTag::Null);
        Node(this_type const &other);
        Node(this_type &&other);
        ~Node();
        this_type &operator=(this_type const &other);
        void swap(this_type &other);

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        int type() const;

        bool is_root() const;

        bool is_leaf() const;

        size_t depth() const;

        bool same_as(this_type const &other) const;

        // attributes
        const Attributes &attributes() const; // return list of attributes
        Attributes &attributes();             // return list of attributes

        //----------------------------------------------------------------------------------------------------------
        // content
        Content &content();
        const Content &content() const;

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        std::shared_ptr<this_type> operator[](T const &v) { return child(v); }

        //----------------------------------------------------------------------------------------------------------
        // level 0
        std::shared_ptr<Node> parent() const; // return parent node
        void remove();                          // remove self from parent

        //----------------------------------------------------------------------------------------------------------
        // level 1
        range select(XPath const &path) const;                       // select from children
        std::shared_ptr<Node> select_one(XPath const &path) const; // return the first selected child

        //----------------------------------------------------------------------------------------------------------
        // level 2
        range ancestor() const;                            // return ancestor
        range descendants() const;                         // return descendants
        range leaves() const;                              // return leave nodes in traversal order
        range slibings() const;                            // return slibings
        range path(this_type const &target) const;         // return the shortest path to target
        ptrdiff_t distance(const this_type &target) const; // lenght of short path to target

    private:
        std::shared_ptr<Node> m_parent_;
        std::unique_ptr<Entry> m_entry_;
    };

    std::ostream &operator<<(std::ostream &os, Node const &d);

} // namespace sp

#endif // SP_NODE_H_