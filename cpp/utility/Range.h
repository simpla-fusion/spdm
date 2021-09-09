#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include "Iterator.h"
#include <memory>
namespace sp
{

//##############################################################################################################
template <typename... T>
class Range;
template <typename T>
class Range<T> : public std::pair<Iterator<T>, Iterator<T>>
{

public:
    typedef Iterator<T> iterator;

    typedef std::pair<iterator, iterator> base_type;
    typedef Range<T> this_type;
    typedef typename iterator::difference_type difference_type;
    typedef typename iterator::pointer pointer;
    typedef typename iterator::value_type value_type;

    using base_type::first;
    using base_type::second;

    Range() {}

    Range(const iterator& first, const iterator& second) : base_type(first, second) {}

    template <typename V, typename... Args>
    Range(const std::shared_ptr<V>& p, difference_type len, Args&&... args) : base_type(Iterator<T>(p), Iterator<T>(p, len)) {}

    template <typename V, typename... Args>
    Range(V const& first, V const& second, Args&&... args)
        : Range{iterator(first, std::forward<Args>(args)...), iterator(second, std::forward<Args>(args)...)} {}

    Range(const base_type& p) : base_type(p) {}

    Range(base_type&& p) : base_type(std::move(p)) {}

    ~Range(){};

    void swap(this_type& other) { base_type::swap(other); }

    size_t size() const { return std::distance(first, second); }

    iterator begin() const { return first; }

    iterator end() const { return second; }

    template <typename U, typename TMapper>
    Range<U>
    map(const TMapper& mapper) const { return Range<U>{first, second, mapper}; };

    template <typename U, typename V, typename... Others, typename TMapper>
    Range<U, V, Others...>
    map(const TMapper& mapper) const { return Range<U, V, Others...>{first, second, mapper}; };

    template <typename TFilter>
    this_type
    filter(const TFilter& filter) const { return this_type{first, second, filter}; };
};
//##############################################################################################################
template <typename U, typename V, typename... Others>
class Range<U, V, Others...> : public Range<std::tuple<U, V, Others...>>
{
public:
    typedef Range<std::tuple<U, V, Others...>> base_type;
    using base_type::Range;
};

} // namespace sp
#endif // SP_RANGE_H_