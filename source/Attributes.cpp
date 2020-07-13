#include "Attributes.h"
#include "utility/Logger.h"

namespace sp
{

class AttributesImpl : public Attributes
{
public:
    AttributesImpl();

    ~AttributesImpl();

    AttributesImpl(const AttributesImpl& other);

    Attributes* copy() const;


    AttributesImpl(AttributesImpl&& other) = delete;
    AttributesImpl operator=(AttributesImpl const& other) = delete;

    Range<Iterator<const std::pair<const std::string, std::any>>> items() const;

    bool has_a(std::string const& key) const;

    bool check(std::string const& key, std::any const& v) const;

    void erase(std::string const& key);

    std::any get_any(std::string const& key) const;

    std::any get_any(std::string const& key, std::any const& default_value);

    void set_any(std::string const& key, std::any const& v);

    std::map<std::string, std::any> m_attributes_;
};

AttributesImpl::AttributesImpl() {}

AttributesImpl::AttributesImpl(const AttributesImpl& other) : m_attributes_(other.m_attributes_) {}

AttributesImpl::~AttributesImpl() {}

Attributes* AttributesImpl::copy() const { return new AttributesImpl(*this); }

bool AttributesImpl::has_a(std::string const& key) const { return m_attributes_.find(key) != m_attributes_.end(); }

bool AttributesImpl::check(std::string const& key, std::any const& v) const
{
    NOT_IMPLEMENTED;
    return has_a(key);
}

void AttributesImpl::erase(std::string const& key) { m_attributes_.erase(m_attributes_.find(key)); }

std::any AttributesImpl::get_any(std::string const& key) const { return m_attributes_.at(key); }

std::any AttributesImpl::get_any(std::string const& key, std::any const& default_value)
{
    return m_attributes_.emplace(key, default_value).first->second;
}

void AttributesImpl::set_any(std::string const& key, std::any const& v) { m_attributes_[key] = v; }

Range<Iterator<const std::pair<const std::string, std::any>>>
AttributesImpl::items() const
{
    return std::move(Range<Iterator<const std::pair<const std::string, std::any>>>{
        Iterator<const std::pair<const std::string, std::any>>{m_attributes_.begin(), [](const auto& it) { return it.operator->(); }},
        Iterator<const std::pair<const std::string, std::any>>{m_attributes_.end()}});
}

Attributes* Attributes::create() { return new AttributesImpl; }

} // namespace sp