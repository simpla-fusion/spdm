#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
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
        class Range;

        SpNode();
        virtual ~SpNode();
        explicit SpNode(SpNode *parent);
        SpNode(SpNode &&other);
        SpNode(SpNode const &);
        SpNode &operator=(SpNode const &) = default;

        std::ostream &repr(std::ostream &os) const;

        void swap(SpNode &other);

        bool is_root() const;

        bool empty() const;

        SpAttribute attribute(std::string const &) const;

        SpAttribute attribute(std::string const &);

        SpRange<SpAttribute> attributes() const;

        SpNode &self() { return *this; }

        SpNode parent() const;

        SpNode next_slibing() const;

        SpNode first_child() const;

        Range children() const;

        Range slibings() const;

        Range select(SpXPath const &path) const;

        void append_child(this_type const &);
        void append_child(this_type &&);
        void remove_child(this_type const &);

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

    std::ostream &operator<<(std::ostream &os, SpNode const &d) { return d.repr(os); }

    class SpNode::Range : public SpRange<SpNode>
    {
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