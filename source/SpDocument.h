#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include "SpUtil.h"
#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace sp
{
    class SpXPath;
    class SpAttribute;
    class SpNode;
    class SpDocument;

    std::ostream &operator<<(std::ostream &os, SpNode const &d);

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
        typedef Iterator<SpNode> iterator;
        typedef Range<SpNode> range;

        SpNode();
        virtual ~SpNode();
        SpNode(iterator parent);
        SpNode(SpNode &&other);
        SpNode(SpNode const &) = delete;
        SpNode &operator=(SpNode const &) = delete;

        void swap(SpNode &other);

        bool is_root() const;

        bool empty() const;

        bool same_as(SpNode const &other) const;

        ptrdiff_t distance(this_type const &other) const;

        Attribute attribute(std::string const &) const;

        Range<Attribute> attributes() const;

        iterator parent() const;

        iterator next() const;

        iterator first_child() const;

        range children() const;

        range slibings() const;

        range select(SpXPath const &path) const;

        std::ostream &repr(std::ostream &os) const;

        void append_child(SpNode const &);
        void append_child(SpNode &&);

    private:
        class Backend;
        Backend *m_pimpl_;
        SpNode *m_parent_;
    };

    SpNode *next(SpNode *);
    SpNode const *next(SpNode const *);

    class SpNode::Attribute
    {
    public:
        Attribute(SpNode const *p = nullptr, std::string const &name = "");
        ~Attribute();
        Attribute(Attribute &&other);

        Attribute(Attribute const &) = delete;
        Attribute &operator=(Attribute const &) = delete;

        std::string name() const;
        std::any value() const;
        bool same_as(Attribute const &) const;
        size_t distance(Attribute const&) const;

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

    SpNode::iterator next(SpNode const &n) { return n.next(); }
    bool same_as(SpNode::Attribute &first, SpNode::Attribute &second) { return first.same_as(second); }
    ptrdiff_t distance(SpNode::Attribute &first, SpNode::Attribute &second) { return first.distance(second); }

    SpNode::Attribute *next(SpNode::Attribute *);
    SpNode::Attribute const *next(SpNode::Attribute const *);

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

        SpNode const &root() const;

        int load(std::string const &);
        int save(std::string const &);
        int load(std::istream const &);
        int save(std::ostream const &);

    private:
        SpNode *m_root_;
    };
} // namespace sp
#endif //SPDB_DOCUMENT_H_