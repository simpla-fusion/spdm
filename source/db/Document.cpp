#include "Document.h"
#include "../utility/Factory.h"
#include "Entry.h"
namespace sp::db
{
Document::OID::OID() : m_id_(reinterpret_cast<unsigned long>(this)) {}
Document::Document() {}
Document::Document(const std::string& uri) {}
Document::Document(const Document&) {}
Document::Document(Document&&) {}
Document::~Document() {}

void Document::swap(Document& other) { m_root_.swap(other.m_root_); }

void Document::load(const std::string& request)
{

    std::shared_ptr<EntryObject> obj;

    obj = ::sp::utility::Factory<EntryObject, Entry*>::create(request, &m_root_);

    if (obj == nullptr)
    {
        RUNTIME_ERROR << "Can not parse request " << request << std::endl;

        throw std::runtime_error("Can not create Entry for scheme: [" + request + "]");
    }
    else
    {
        obj->fetch(request);

        m_root_.emplace<Entry::type_tags::Object>(obj);

        VERBOSE << "Load Entry Object plugin:" << request << std::endl;
    }
}

} // namespace sp::db