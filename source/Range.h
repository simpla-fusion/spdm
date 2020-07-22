#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include "Iterator.h"
#include <memory>
namespace sp
{

//##############################################################################################################
template <typename... T>
class Range : public std::pair<Iterator<T...>, Iterator<T...>>
{

public:
    typedef Iterator<T...> iterator;
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

    size_t size() const { return std::distance(first, second); }

    iterator begin() const { return first; }

    iterator end() const { return second; }

    template <typename U, typename V, typename... Others, typename TFun>
    Range<U, V, Others...>
    map(const TFun& fun) const
    {
        return Range<U, V, Others...>{Iterator<U, V, Others...>(begin(), fun), Iterator<U, V, Others...>(end(), fun)};
    };

    template <typename U, typename TFun>
    Range<U> map(const TFun& fun) const
    {
        // Iterator<U>(begin(), fun), Iterator<U>(end(), fun)
        // NOT_IMPLEMENTED;
        return Range<U>{};
    };
};
//##############################################################################################################

} // namespace sp
#endif // SP_RANGE_H_