#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include "Iterator.h"
#include <memory>
namespace sp
{

//##############################################################################################################
template <typename IT>
class Range : public std::pair<IT, IT>
{

public:
    typedef IT iterator;
    typedef typename iterator::pointer pointer;
    typedef typename std::pair<iterator, iterator> base_type;
    using base_type::first;
    using base_type::second;

    Range() {}

    Range(iterator const& first, iterator const& second) : base_type(first, second) {}

    template <typename U0, typename U1>
    Range(U0 const& first, U1 const& second) : Range(iterator(first), iterator(second)) {}

    template <typename U0, typename U1>
    Range(std::pair<U0, U1>&& r) : Range(std::move(std::get<0>(r)), std::move(std::get<1>(r))) {}

    Range(base_type const& p) : base_type(p) {}

    ~Range(){};

    iterator begin() const { return first; }

    iterator end() const { return second; }
};
//##############################################################################################################

} // namespace sp
#endif // SP_RANGE_H_