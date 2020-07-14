#include "Attributes.h"
#include "utility/Logger.h"

namespace sp
{

Attributes::Attributes() {}

Attributes::Attributes(const Attributes& other) : m_data_(other.m_data_) {}

Attributes::~Attributes() {}

Attributes* Attributes::copy() const { return new Attributes(*this); }

bool Attributes::has_a(std::string const& key) const { return m_data_.find(key) != m_data_.end(); }

bool Attributes::check(std::string const& key, std::any const& v) const
{
    NOT_IMPLEMENTED;
    return has_a(key);
}

void Attributes::erase(std::string const& key) { m_data_.erase(m_data_.find(key)); }

std::any Attributes::get(std::string const& key) const { return m_data_.at(key); }

std::any Attributes::get(std::string const& key, std::any const& default_value)
{
    return m_data_.emplace(key, default_value).first->second;
}

void Attributes::set(std::string const& key, std::any const& v) { m_data_[key] = v; }

void Attributes::clear() { m_data_.clear(); }

Range<Iterator<const std::pair<const std::string, std::any>>>
Attributes::items() const
{
    return std::move(Range<Iterator<const std::pair<const std::string, std::any>>>{
        Iterator<const std::pair<const std::string, std::any>>{m_data_.begin(), [](const auto& it) { return it.operator->(); }},
        Iterator<const std::pair<const std::string, std::any>>{m_data_.end()}});
}
Range<Iterator<std::pair<const std::string, std::any>>>
Attributes::items()
{
    return std::move(Range<Iterator<std::pair<const std::string, std::any>>>{
        Iterator<std::pair<const std::string, std::any>>{m_data_.begin(), [](const auto& it) { return it.operator->(); }},
        Iterator<std::pair<const std::string, std::any>>{m_data_.end()}});
}
// Attributes* Attributes::create() { return new Attributes; }

} // namespace sp