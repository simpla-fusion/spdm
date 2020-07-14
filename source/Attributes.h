#ifndef SP_ATTRIBUTES_H_
#define SP_ATTRIBUTES_H_
#include "Range.h"
#include <any>
#include <map>
#include <memory>
#include <ostream>
#include <string>
namespace sp
{

class Attributes
{
public:
    Attributes();
    Attributes(const Attributes& other);
    Attributes(Attributes&& other);
    ~Attributes();

    void swap(Attributes& other);

    Attributes& operator=(Attributes const& other);

    Attributes* copy() const;

    Range<Iterator<const std::pair<const std::string, std::any>>> items() const;
    
    Range<Iterator<std::pair<const std::string, std::any>>> items();

    void clear();

    bool has_a(std::string const& key) const;

    bool check(std::string const& key, const std::any& v) const;

    void erase(std::string const& key);

    std::any get(std::string const& key) const;

    std::any get(std::string const& key, const std::any& default_value);

    void set(std::string const& key, const std::any& v);

    template <typename U, typename V>
    void set(std::string const& key, V const& v) { set(std::make_any<U>(v)); }

    template <typename U>
    U get(std::string const& key) const { return std::any_cast<U>(get(key)); }

    template <typename U, typename V>
    U get(std::string const& key, V const& default_value) { return std::any_cast<U>(get(key, std::make_any<U>(default_value))); }

private:
    std::map<std::string, std::any> m_data_;
};

} // namespace sp
#endif //SP_ATTRIBUTES_H_