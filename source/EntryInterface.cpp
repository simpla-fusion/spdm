#include "EntryInterface.h"
#include "utility/Factory.h"

namespace sp
{
EntryInterface::EntryInterface(Entry* self, const std::string& name, Entry* parent)
    : m_self_(self), m_name_(name), m_parent_(parent) {}

EntryInterface::EntryInterface(const EntryInterface& other)
    : m_self_(other.m_self_),
      m_name_(other.m_name_),
      m_parent_(other.m_parent_)
{
}

EntryInterface::EntryInterface(EntryInterface&& other)
    : m_self_(other.m_self_),
      m_name_(other.m_name_),
      m_parent_(other.m_parent_)
{
}

std::string EntryInterface::prefix() const { return m_parent_ == nullptr ? m_name_ : m_parent_->prefix() + "/" + m_name_; };

std::string EntryInterface::name() const { return m_name_; };

Entry::iterator EntryInterface::parent() const { return Entry::iterator(m_parent_); };

std::unique_ptr<EntryInterface> EntryInterface::create(const std::string& backend, Entry* self, const std::string& name, Entry* parent)
{
    return Factory<EntryInterface, Entry*, const std::string&, Entry*>::create(backend, self, name, parent);
}

} // namespace sp