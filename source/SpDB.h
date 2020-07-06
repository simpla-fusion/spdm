#ifndef SPDB_H_
#define SPDB_H_
#include "SpUtil.h"
#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>
#include <vector>

namespace sp
{
    class SpXPath;
    class SpNode;
    class SpDocument;
    class SpDB;

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

    template <typename T0, typename T1 = T0>
    class SpRange : std::pair<T0, T1>
    {
    };

    class SpNode
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

        typedef SpNode this_type;
        typedef SpNode node_type;
        typedef SpRange<iterator> range_type;

        explicit SpNode(std::shared_ptr<SpNode> const &parent = nullptr);
        virtual ~SpNode();
        SpNode(SpNode &&other);
        SpNode(SpNode const &);
        SpNode &operator=(SpNode const &) = default;
        void swap(SpNode &other);
        std::ostream &repr(std::ostream &os) const; // represent object as string and push ostream

        std::map<std::string, std::any> attributes() const;        // return list of attributes
        std::any attribute(std::string const &name) const;         // get attribute, return nullptr is name does not exist
        int attribute(std::string const &name, std::any const &v); // set attribute
        int remove_attribute(std::string const &name);             // remove attribuet

        void value(std::any const &); // set value
        std::any value() const;       // get value

        bool same_as(this_type const &other) const;
        bool empty() const;
        size_t size() const;

        TypeOfNode type() const;

        bool is_null() const { return type() == TypeOfNode::Null; }
        bool is_scalar() const { return type() == TypeOfNode::Scalar; }
        bool is_block() const { return type() == TypeOfNode::Block; }
        bool is_list() const { return type() == TypeOfNode::List; }
        bool is_object() const { return type() == TypeOfNode::Object; }
        //----------------------------------------------------------------------------------------------------------
        // as Hierarchy tree node
        template <typename T>
        SpNode operator[](T const &v) { return child(v); }

        bool is_root() const; // parent().empty() is true
        bool is_leaf() const; // children().size() =0
        size_t depth() const; // distance(root())

        //----------------------------------------------------------------------------------------------------------
        // level 0
        void remove(); // remove self

        iterator parent() const;      // return parent node
        iterator first_child() const; // return first child node
        range_type children() const;  // return children

        // as object
        iterator child(std::string const &) const; // return node at key,  if key does not exist then throw exception
        iterator child(std::string const &);       // return node at key,  if key does not exist then create one
        int remove_child(std::string const &key);  // remove child with key, return true if key exists

        // as list
        iterator child(int);             // return node at idx,  if idx >size() then throw exception
        iterator child(int) const;       // return node at idx,  if idx >size() then throw exception
        iterator insert_before(int pos); // insert new child node before pos, return new node
        iterator insert_after(int pos);  // insert new child node after pos, return new node
        iterator prepend();              // insert new child node before first child
        iterator append();               // insert new child node afater last child
        int remove_child(int idx);       // remove child at pos, return true if idx exists

        //----------------------------------------------------------------------------------------------------------
        // level 1
        range_type select(SpXPath const &path) const;   // select from children
        iterator select_one(SpXPath const &path) const; // return the first selected child

        //----------------------------------------------------------------------------------------------------------
        // level 2
        range_type ancestor() const;                       // return ancestor
        range_type descendants() const;                    // return descendants
        range_type leaves() const;                         // return leave nodes in traversal order
        range_type slibings() const;                       // return slibings
        range_type path(SpNode const &target) const;       // return the shortest path to target
        ptrdiff_t distance(const this_type &target) const; // lenght of short path to target

    private:
        std::shared_ptr<SpNode> m_parent_;
        struct pimpl_s;
        std::unique_ptr<pimpl_s> m_pimpl_;
    };

    std::ostream &operator<<(std::ostream &os, SpNode const &d) { return d.repr(os); }
    //----------------------------------------------------------------------------------------------------------
    class SpNode::iterator : public std::iterator<std::input_iterator_tag, SpNode>
    {

    public:
        iterator(std::shared_ptr<SpNode> const &p = nullptr);
        virtual ~iterator();
        iterator(iterator const &);
        iterator(iterator &&);
        iterator &swap(iterator &other);

        iterator &operator=(iterator const &other) { return iterator(other).swap(*this); }

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

    class SpDocument
    {
    public:
        class OID
        {
        public:
            OID();
            ~OID() = default;

            OID(unsigned long id);

            OID(OID &&) = default;
            OID(OID const &) = default;
            OID &operator=(OID const &) = default;

            operator unsigned long() const { return m_id_; }
            unsigned long id() const { return m_id_; }

            bool operator==(OID const &other) { return m_id_ == other.m_id_; }

        private:
            unsigned long m_id_ = 0;
        };

        typedef OID id_type;

        OID oid;

        SpDocument();

        SpDocument(SpDocument &&);

        ~SpDocument();

        SpDocument(SpDocument const &) = delete;
        SpDocument &operator=(SpDocument const &) = delete;

        void schema(SpDocument const &schema);
        const SpDocument &schema();
        void schema(std::string const &schema);
        const std::string &schema_id();

        const SpNode &root() const;
        SpNode &root();

        int load(std::string const &);
        int save(std::string const &);
        int load(std::istream const &);
        int save(std::ostream const &);

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    class SpDB
    {
    public:
        SpDB();
        ~SpDB();

        int connect(std::string const &connection, std::string const &schema = "");
        int disconnect();

        SpDocument create(SpDocument::id_type const &oid);
        SpDocument open(SpDocument::id_type const &oid);
        int insert(SpDocument::id_type const &oid, SpDocument &&);
        int insert(SpDocument::id_type const &oid, SpDocument const &);
        int remove(SpDocument::id_type const &oid);
        int remove(std::string const &query);

        std::vector<SpDocument> search(std::string const &query);

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

} // namespace sp

#endif //SPDB_H_