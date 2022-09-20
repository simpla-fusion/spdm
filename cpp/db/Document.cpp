#include "Document.hpp"
#include "../utility/Factory.hpp"
#include "Entry.hpp"

namespace sp::db
{
Document::OID::OID() : m_id_(reinterpret_cast<unsigned long>(this)) {}
Document::Document() {}
Document::Document(const std::string& uri) { load(uri); }
Document::Document(const Document& other) : m_root_(other.m_root_), m_schema_(other.m_schema_) {}
Document::Document(Document&& other) : m_root_(std::move(other.m_root_)), m_schema_(std::move(other.m_schema_)) {}
Document::~Document() {}

void Document::swap(Document& other) { std::swap(m_root_, other.m_root_); }

void Document::load(const std::string& request)
{
    //  m_root_.template emplace<URI>(request);
}

} // namespace sp::db