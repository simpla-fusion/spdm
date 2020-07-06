#ifndef SP_NODE_H_
#define SP_NODE_H_
#include "SpNode.h"
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

    template <typename TAG = nullptr_t>
    class SpNode;
    typedef SpNode<> node_t;

    template <typename TAG = nullptr_t>
    class Attributes;
    typedef Attributes<> attributes_t;

    template <typename TAG = nullptr_t, NodeTag tag = NodeTag::Null>
    class Content;
    typedef Content<> content_t;

    node_t *create_node(std::shared_ptr<node_t> const &parent = nullptr, NodeTag tag = NodeTag::Null);

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

    template <typename TAG>
    class Attributes
    {
    public:
        Attributes() = default;
        virtual ~Attributes() = default;
        Attributes(Attributes const &) = default;
        Attributes(Attributes &&) = default;

        Attributes &operator=(Attributes const &other)
        {
            Attributes(other).swap(*this);
            return *this;
        };
        void swap(Attributes &) {}
        Attributes *copy() const { return new Attributes(*this); }

        std::any get(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int set(std::string const &name, std::any const &v); // set attribute
        int remove(std::string const &name);                 // remove attribuet
    };

    template <>
    class Content<>
    {
    public:
        typedef content_t this_type;
        Content() = default;
        virtual ~Content() = default;
        Content(Content const &) = default;
        Content(Content &&) = default;
        Content &operator=(Content const &other)
        {
            Content(other).swap(*this);
            return *this;
        };

        void swap(Content &);

        Content *as(NodeTag tag);
        static Content *create(NodeTag tag = NodeTag::Null);

        Content *copy() const { return new Content(*this); }
        NodeTag type() const { return NodeTag::Null; }
    };

    template <typename TAG>
    class Content<TAG> : public Content<>
    {
    };

    template <typename TAG>
    class Content<TAG, NodeTag::Scalar> : public Content<TAG>
    {
    public:
        void swap(Content<TAG, NodeTag::Scalar> &other);
        std::any value();
        void value(std::any const &v);
    };

    template <typename TAG>
    class Content<TAG, NodeTag::Block> : public Content<TAG>
    {
    public:
        void swap(Content<TAG, NodeTag::Block> &other);

        void data(std::shared_ptr<char> const &);
        std::shared_ptr<char> data();
        std::shared_ptr<char> data() const;

        void dims(std::vector<size_t> const &);
        const std::vector<size_t> &dims() const;

        void element_size(size_t s);
        size_t element_size() const;

        template <typename V>
        void data(std::shared_ptr<V> const &v, int nd, size_t const *d)
        {
            data(std::reinterpret_pointer_cast<char>(v));
            element_size(sizeof(V));
            std::vector<size_t> t_dims(d, d + nd);
            this->dims(t_dims);
        }

        template <typename V>
        std::shared_ptr<V> data() { return std::reinterpret_pointer_cast<char>(data()); }
    };

    template <typename TAG>
    class Content<TAG, NodeTag::List> : public Content<TAG>
    {
    public:
        void swap(Content<TAG, NodeTag::List> &other);

        size_t size() const;
        Range<Iterator<SpNode<TAG>>, Iterator<SpNode<TAG>>> children(); // return children
        std::shared_ptr<node_t> child(int);                             // return node at idx,  if idx >size() then throw exception
        std::shared_ptr<node_t> append();                               // insert new child node after last child
        Iterator<SpNode<TAG>> first_child();                            // return first child node, next(first_child()) return the nearest slibing
        int remove_child(int idx);                                      // remove child with key, return true if key exists
    };

    template <typename TAG>
    class Content<TAG, NodeTag::Object> : public Content<TAG>
    {
    public:
        void swap(Content<TAG, NodeTag::Object> &other);
        size_t size() const;
        Range<Iterator<SpNode<TAG>>, Iterator<SpNode<TAG>>> children(); // return children
        std::shared_ptr<SpNode<TAG>> find(std::string const &);         // return node at key,  if key does not exist then return nullptr
        std::shared_ptr<SpNode<TAG>> child(std::string const &);        // return node at key,  if key does not exist then create one
        int remove_child(std::string const &key);                       // remove child with key, return true if key exists
    };
    //----------------------------------------------------------------------------------------------------------

    template <typename TAG>
    class SpNode : public std::enable_shared_from_this<SpNode<TAG>>
    {
    public:
        typedef SpNode<TAG> this_type;
        typedef Iterator<std::shared_ptr<this_type>> iterator;
        typedef Range<iterator, iterator> range;

        // SpNode(std::shared_ptr<node_t> const &parent = nullptr, NodeTag tag = NodeTag::Null);
        // SpNode(this_type const &other);
        // SpNode(this_type &&other);
        // ~SpNode();
        // this_type &operator=(this_type const &other);
        // void swap(this_type &other);

        SpNode(std::shared_ptr<node_t> const &parent)
            : m_parent_(parent),
              m_attributes_(new Attributes<TAG>()),
              m_content_(new Content<TAG>()){};

        SpNode(SpNode const &other)
            : m_parent_(other.m_parent_),
              m_content_(other.m_content_->copy()),
              m_attributes_(other.m_attributes_->copy()) {}

        SpNode(SpNode &&other)
            : m_parent_(other.m_parent_),
              m_content_(std::move(other.m_content_)),
              m_attributes_(std::move(other.m_attributes_))
        {
            other.m_attributes_.reset();
            other.m_content_.reset();
        }

        ~SpNode() {}

        this_type &operator=(this_type const &other)
        {
            this_type(other).swap(*this);
            return *this;
        }

        void swap(this_type &other)
        {
            std::swap(m_parent_, other.m_parent_);
            std::swap(m_content_, other.m_content_);
            std::swap(m_attributes_, other.m_attributes_);
        }

        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        NodeTag type() const { return m_content_->type(); }

        bool is_root() const { return m_parent_ == nullptr; }

        bool is_leaf() const { return !(type() == NodeTag::List || type() == NodeTag::Object); }

        size_t depth() const { return m_parent_ == nullptr ? 0 : m_parent_->depth() + 1; }

        bool same_as(this_type const &other) const { return this == &other; }

        // template <NodeTag C>
        // auto &as() { return *dynamic_cast<Content<C, TAG> *>(m_content_.get()); }

        // template <NodeTag C>
        // auto const &as() const { return *dynamic_cast<Content<C, TAG> const *>(m_content_->as(TAG)); }

        // Content<TAG, NodeTag::Scalar> &as_scalar() { return as<NodeTag::Scalar>(); }
        // const Content<TAG, NodeTag::Scalar> &as_scalar() const { return as<NodeTag::Scalar>(); }

        // Content<TAG, NodeTag::Block> &as_block() { return as<NodeTag::Block>(); }
        // const Content<TAG, NodeTag::Block> &as_block() const { return as<NodeTag::Block>(); }

        // Content<TAG, NodeTag::List> &as_list() { return as<NodeTag::List>(); }
        // const Content<TAG, NodeTag::List> &as_list() const { return as<NodeTag::List>(); }

        // Content<TAG, NodeTag::Object> &as_object() { return as<NodeTag::Object>(); }
        // const Content<TAG, NodeTag::Object> &as_object() const { return as<NodeTag::Object>(); }

        // attributes
        const attributes_t &attributes() const { return *m_attributes_; } // return list of attributes
        attributes_t &attributes() { return *m_attributes_; }             // return list of attributes

        //----------------------------------------------------------------------------------------------------------
        // content
        content_t &content() { return *m_content_; }
        const content_t &content() const { return *m_content_; }

        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        std::shared_ptr<this_type> operator[](T const &v) { return child(v); }

        //----------------------------------------------------------------------------------------------------------
        // level 0
        std::shared_ptr<node_t> parent() const; // return parent node
        void remove();                          // remove self from parent

        //----------------------------------------------------------------------------------------------------------
        // level 1
        range select(SpXPath const &path) const;                       // select from children
        std::shared_ptr<node_t> select_one(SpXPath const &path) const; // return the first selected child

        //----------------------------------------------------------------------------------------------------------
        // level 2
        range ancestor() const;                            // return ancestor
        range descendants() const;                         // return descendants
        range leaves() const;                              // return leave nodes in traversal order
        range slibings() const;                            // return slibings
        range path(node_t const &target) const;            // return the shortest path to target
        ptrdiff_t distance(const this_type &target) const; // lenght of short path to target

    protected:
        SpNode(std::shared_ptr<node_t> const &parent, Attributes<TAG> *attr, Content<TAG> *content)
            : m_parent_(parent),
              m_attributes_(attr),
              m_content_(content){};

    private:
        std::shared_ptr<node_t> m_parent_;
        std::unique_ptr<Content<TAG>> m_content_;
        std::unique_ptr<Attributes<TAG>> m_attributes_;
        ;
    };

    template <typename TAG>
    std::ostream &operator<<(std::ostream &os, SpNode<TAG> const &d) { return d.repr(os); }

} // namespace sp

#endif // SP_NODE_H_