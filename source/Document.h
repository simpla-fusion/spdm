#ifndef SP_DOCUMENT_H_
#define SP_DOCUMENT_H_
#include "Entry.h"
#include <iostream>
#include <memory>
#include <string>

namespace sp
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

    Document();

    Document(Document&&);

    ~Document();

    Document(Document const&) = delete;
    Document& operator=(Document const&) = delete;

    void schema(Document const& schema);
    const Document& schema();
    void schema(const std::string& schema);
    const std::string& schema_id();

    const Entry& root() const;
    Entry& root();

    int load(std::string const&);
    int save(std::string const&);
    int load(std::istream const&);
    int save(std::ostream const&);

private:
    std::unique_ptr<Entry> m_pimpl_;
};

} // namespace sp

#endif // SP_DOCUMENT_H_