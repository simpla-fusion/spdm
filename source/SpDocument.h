#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include "SpUtil.h"
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
namespace sp
{
    class SpXPath;
    class SpAttribute;
    class SpNode;
    class SpDocument;

    template <typename T, typename... Others>
    class SpIterator
    {
    public:
        typedef SpIterator<T, Others...> this_type;
        typedef T value_type;
        typedef value_type *pointer;
        typedef value_type &reference;

        SpIterator(pointer d = nullptr) : m_self_(d){};
        virtual ~SpIterator() = default;
        SpIterator(this_type const &) = default;
        SpIterator(this_type &&) = default;

        this_type &operator=(this_type const &) = default;

        bool operator==(this_type const &other) const { return equal(other); }
        bool operator!=(this_type const &other) const { return !equal(other); }
        ptrdiff_t operator-(this_type const &other) const { return distance(other); }

        reference operator*() const { return *m_self_; };
        pointer operator->() const { return m_self_; };

        this_type operator++(int)
        {
            this_type res(*this);
            m_self_ = next(m_self_);
            return res;
        }

        this_type &operator++()
        {
            m_self_ = next(m_self_);
            return *this;
        }

        pointer next(pointer p) const { return std::next(p); }
        bool equal(this_type const &other) const { return *m_self_ == *other.m_self_; }
        bool distance(this_type const &other) const { return std::distance(m_self_, other.m_self_); }

    private:
        pointer m_self_;
    };

    template <typename T>
    class SpRange : public std::pair<SpIterator<T>, SpIterator<T>>
    {

    public:
        typedef std::pair<SpIterator<T>, SpIterator<T>> base_type;
        typedef SpRange<T> this_type;
        typedef SpIterator<T> iterator;
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

    class SpNode
    {

    public:
        typedef SpNode this_type;

        class Attribute;
        friend class Attribute;
        class Iterator;
        class Range;

        SpNode();
        virtual ~SpNode();
        explicit SpNode(SpNode *parent);
        SpNode(SpNode &&other);
        SpNode(SpNode const &) = delete;
        SpNode &operator=(SpNode const &) = delete;

        bool operator==(SpNode const &) const;

        void swap(SpNode &other);

        bool is_root() const;

        bool empty() const;

        Attribute attribute(std::string const &) const;

        SpRange<Attribute> attributes() const;

        Iterator parent() const;

        Iterator next() const;

        Iterator first_child() const;

        Range children() const;

        Range slibings() const;

        Range select(SpXPath const &path) const;

        std::ostream &repr(std::ostream &os) const;

        void append_child(SpNode const &);
        void append_child(SpNode &&);

    private:
        struct pimpl_s;
        pimpl_s *m_pimpl_;
        SpNode *m_parent_;
    };

    std::ostream &operator<<(std::ostream &os, SpNode const &d) { return d.repr(os); }

    class SpNode::Iterator : public SpIterator<SpNode>
    {
    public:
        typedef SpIterator<SpNode> base_type;
        typedef Iterator this_type;

        using typename base_type::pointer;
        using typename base_type::reference;
        using typename base_type::value_type;

        Iterator(pointer d = nullptr) : base_type(d){};
        ~Iterator() = default;
        Iterator(this_type const &) = default;
        Iterator(this_type &&) = default;

        this_type &operator=(this_type const &) = default;

        pointer next(pointer p);
        bool equal(this_type const &other);
        bool distance(this_type const &other);
    };

    class SpNode::Range : public SpRange<SpNode::Iterator>
    {
    };

    class SpNode::Attribute
    {
    public:
        class Iterator;
        class Range;

        Attribute(SpNode const *p = nullptr, std::string const &name = "");
        ~Attribute();
        Attribute(Attribute const &);
        Attribute(Attribute &&other);

        Attribute &operator=(Attribute const &) = delete;
        bool operator==(Attribute const &other) const { return same_as(other); };

        std::string name() const;
        std::any value() const;
        bool same_as(Attribute const &) const;
        size_t distance(Attribute const &) const;

        std::any get() const;
        void set(std::any const &);

        template <typename T>
        void set(T const &v) { this->set(std::any(v)); };

        template <typename T>
        Attribute &operator=(T const &v)
        {
            this->set(v);
            return *this;
        }

        template <typename T>
        T as() const { return std::any_cast<T>(this->get()); }

    private:
        SpNode const *m_node_;
        std::string m_name_;
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

        SpNode::Iterator root() const;

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