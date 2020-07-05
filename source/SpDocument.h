#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include "Array.h"
#include "SpDataBlock.h"
#include "SpUtil.h"
#include <algorithm>
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace sp
{
    class SpXPath;
    class SpAttribute;
    class SpNode;
    class SpDocument;

    template <typename T>
    class SpRange : public std::pair<T, T>
    {

    public:
        typedef std::pair<T, T> base_type;
        typedef SpRange<T> this_type;
        typedef T iterator;
        using base_type::first;
        using base_type::second;

        SpRange(iterator const &a, iterator const &b) : base_type(a, b) {}
        SpRange() = default;
        virtual ~SpRange() = default;
        SpRange(SpRange const &) = default;
        SpRange(SpRange &&) = default;

        size_t size() const { return std::distance(first, second); }
        size_t empty() const { return first == second; }

        auto &begin() const { return base_type::first; }
        auto &end() const { return base_type::second; }
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

        std::string const &value() const;

        SpXPath operator/(std::string const &suffix) const;
        operator std::string() const;

    private:
        std::string m_path_;
    };

    class SpAttribute
    {
    public:
        typedef SpAttribute this_type;
        typedef SpNode parent_type;

        SpAttribute(parent_type const *p = nullptr, std::string const &name = "");
        ~SpAttribute();
        SpAttribute(this_type const &);
        SpAttribute(this_type &&other);
        void swap(this_type &other);

        std::ostream &repr(std::ostream &os) const;

        this_type &operator=(this_type const &other) { return set(other.get()); }

        template <typename T>
        this_type &operator=(T const &v) { return this->set(v); }

        bool same_as(this_type const &) const;
        bool equal(std::any const &) const;

        std::string name() const;
        std::any value() const;

        std::any get() const;
        this_type &set(std::any const &);

        template <typename T>
        this_type &set(T const &v) { return this->set(std::any(v)); };

        template <typename T>
        T as() const { return std::any_cast<T>(this->get()); }

        bool operator==(this_type const &other) const { return equal(other); }
        bool operator!=(this_type const &other) const { return !equal(other); }
        ptrdiff_t operator-(this_type const &other) const { return distance(other); }

        this_type &operator++()
        {
            next();
            return *this;
        }
        this_type &operator*() { return *this; }

        void next();
        bool equal(this_type const &other) const;
        bool distance(this_type const &other) const;

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    std::ostream &operator<<(std::ostream &os, SpAttribute const &d) { return d.repr(os); }

    class SpNode
    {

    public:
        typedef SpNode this_type;

        friend class SpAttribute;

        SpNode();
        virtual ~SpNode();
        explicit SpNode(SpNode *parent);
        SpNode(SpNode &&other);
        SpNode(SpNode const &);
        SpNode &operator=(SpNode const &) = default;
        void swap(SpNode &other);

        std::ostream &repr(std::ostream &os) const;

        SpAttribute attribute(std::string const &) const;
        SpAttribute attribute(std::string const &);
        SpRange<SpAttribute> attributes() const;

        void value(std::any const &);
        std::any value() const;

        template <typename T>
        void value(T const &v) { value(std::any(v)); };
        template <typename T>
        T as() { return std::any_cast<T>(value()); }

        void data(SpDataBlock const &block);
        SpDataBlock data() const;

        template <typename T>
        Array<T> as_array() { return Array<T>(data()); }

        bool equal(this_type const &other) const;
        bool empty() const;
        size_t size() const;
        bool is_null() const;
        bool is_sclar() const;
        bool is_array() const;
        bool is_list() const;
        bool is_object() const;

        // as Hierarchy tree node
        template <typename T>
        SpNode operator[](T const &v) { return child(v); }

        bool is_root() const;                         // parent().empty() is true
        bool is_leaf() const;                         // children().size() =0
        bool distance(this_type const &target) const; // lenght of short path to target
        size_t depth() const;                         // distance(root())

        void remove(); // remove self

        SpNode &self() { return *this; }             // return self
        const SpNode &self() const { return *this; } // return self

        SpNode next() const;                             // return next slibing
        SpNode parent() const;                           // return parent node
        SpNode first_child() const;                      // return first child node
        SpRange<SpNode> ancestor() const;                // return ancestor
        SpRange<SpNode> descendants() const;             // return descendants
        SpRange<SpNode> leaves() const;                  // return leave nodes in traversal order
        SpRange<SpNode> children() const;                // return children
        SpRange<SpNode> slibings() const;                // return slibings
        SpRange<SpNode> path(SpNode const target) const; // return the shortest path to target

        SpRange<SpNode> select(SpXPath const &path) const; // select from children
        SpNode select_one(SpXPath const &path) const;      // return the first selected child

        // as object
        SpNode child(std::string const &) const;  // return node at key,  if key does not exist then throw exception
        SpNode child(std::string const &);        // return node at key,  if key does not exist then create one
        int remove_child(std::string const &key); // remove child with key, return true if key exists

        // as list
        SpNode child(int);                            // return node at idx,  if idx >size() then throw exception
        SpNode child(int) const;                      // return node at idx,  if idx >size() then throw exception
        SpNode insert_before(int pos);                // insert new child node before pos, return new node
        SpNode insert_after(int pos);                 // insert new child node after pos, return new node
        SpNode prepend() { return insert_before(0); } // insert new child node before first child
        SpNode append() { return insert_after(-1); }  // insert new child node afater last child
        int remove_child(int idx);                    // remove child at pos, return true if idx exists

        // as iterator

        bool operator==(this_type const &other) const { return equal(other); }
        bool operator!=(this_type const &other) const { return !equal(other); }

        this_type &operator++()
        {
            next().swap(*this);
            return *this;
        }
        this_type operator++(int)
        {
            SpNode res(*this);
            next().swap(*this);
            return res;
        }
        this_type &operator*() { return *this; }
        this_type *operator->() { return this; }

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
    };

    std::ostream &operator<<(std::ostream &os, SpNode const &d) { return d.repr(os); }

    template <>
    class SpRange<SpNode> : public std::pair<SpNode, SpNode>
    {
    public:
        typedef std::pair<SpNode, SpNode> base_type;
        typedef SpRange<SpNode> this_type;

        typedef SpNode iterator;
        using base_type::first;
        using base_type::second;

        SpRange(iterator const &a, iterator const &b) : base_type(a, b) {}
        SpRange() = default;
        virtual ~SpRange() = default;
        SpRange(SpRange const &) = default;
        SpRange(SpRange &&) = default;

        size_t empty() const { return first == second; }

        size_t size() const;
        SpNode begin() const;
        SpNode end() const;

        typedef std::function<bool(SpNode const &)> filter_type;
        this_type filter(filter_type const &);

        template <typename U>
        SpRange<U> map(std::function<U(SpNode const &)> const &);
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
} // namespace sp
#endif //SPDB_DOCUMENT_H_