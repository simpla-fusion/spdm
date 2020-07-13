#ifndef SP_DOCUMENT_H_
#define SP_DOCUMENT_H_
#include "Node.h"

namespace sp
{
class SpDocument
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

    SpDocument();

    SpDocument(SpDocument&&);

    ~SpDocument();

    SpDocument(SpDocument const&) = delete;
    SpDocument& operator=(SpDocument const&) = delete;

    void schema(SpDocument const& schema);
    const SpDocument& schema();
    void schema(std::string const& schema);
    const std::string& schema_id();

    const SpNode& root() const;
    SpNode& root();

    int load(std::string const&);
    int save(std::string const&);
    int load(std::istream const&);
    int save(std::ostream const&);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

} // namespace sp

#endif // SP_DOCUMENT_H_