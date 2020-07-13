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

    virtual ~Attributes();

    Attributes* copy() const;
    static Attributes* create();

    Attributes(const Attributes& other) = delete;
    Attributes(Attributes&& other) = delete;
    Attributes& operator=(Attributes const& other) = delete;

    virtual Range<Iterator<const std::pair<const std::string, std::any>>> items() const = 0;

    virtual bool has_a(std::string const& key) const = 0;

    virtual bool check(std::string const& key, std::any const& v) const = 0;

    virtual void erase(std::string const& key) = 0;

    virtual std::any get_any(std::string const& key) const = 0;

    virtual std::any get_any(std::string const& key, std::any const& default_value) = 0;

    virtual void set_any(std::string const& key, std::any const& v) = 0;

    template <typename U, typename V>
    void set(std::string const& key, V const& v) { set_any(std::make_any<U>(v)); }

    template <typename U>
    U get(std::string const& key) const { return std::any_cast<U>(get_any(key)); }

    template <typename U, typename V>
    U get(std::string const& key, V const& default_value) { return std::any_cast<U>(get_any(key, std::make_any<U>(default_value))); }
};

} // namespace sp
#endif //SP_ATTRIBUTES_H_