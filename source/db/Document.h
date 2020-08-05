#ifndef SPDB_DOCUMENT_H_
#define SPDB_DOCUMENT_H_
#include "Entry.h"
#include "Schema.h"
#include <iostream>
#include <memory>
#include <string>

namespace sp::db
{

class Document
{
public:
    class OID
    {
    public:
        OID();
        ~OID() = default;

        OID(unsigned long id);

        OID(OID&&) = default;
        OID(OID const&) = default;
        OID& operator=(OID const&) = default;

        operator unsigned long() const { return m_id_; }
        unsigned long id() const { return m_id_; }

        bool operator==(OID const& other) { return m_id_ == other.m_id_; }

    private:
        unsigned long m_id_ = 0;
    };

    typedef OID id_type;

    OID oid;

    typedef Document this_type;

    Document();
    Document(const std::string& uri);
    Document(const Document&);
    Document(Document&&);
    ~Document();

    void swap(Document& other);

    Document& operator=(Document const& other)
    {
        this_type(other).swap(*this);
        return *this;
    };

    void load(const std::string&);
    void save(const std::string&);
    void load(const std::istream&);
    void save(const std::ostream&);

    bool is_writable() const;
    bool is_readable() const;

    const Entry& root() const { return m_root_; }
    Entry& root() { return m_root_; }

    const Schema& schema() const;
    Schema& schema();

    bool validate(const XPath&) const;

    bool check(const XPath&) const;

private:
    Entry m_root_;
    Schema m_schema_;
};

} // namespace sp::db

#endif // SPDB_DOCUMENT_H_