#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include <stddef.h>
#include <iterator>
#include <functional>

template <typename BaseIterator>
class filtered_iterator
{
public:
    typedef filtered_iterator<BaseIterator> this_type;

    typedef std::remove_reference_t<decltype(*std::declval<BaseIterator>())> value_type;

    typedef value_type *pointer;

    typedef value_type &reference;

    typedef BaseIterator base_iterator_type;

    typedef std::function<bool(const value_type &)> filter_type;

    filtered_iterator() = default;
    ~filtered_iterator() = default;
    filtered_iterator(filtered_iterator const &) = default;
    filtered_iterator(filtered_iterator &&) = default;

    filtered_iterator &operator=(filtered_iterator const &) = default;

    template <typename Filter>
    filtered_iterator(base_iterator_type const &b, base_iterator_type const &e, Filter const &filter = {})
        : m_begin_(b), m_end_(e), m_filter_(filter)
    {
        while (m_begin_ != m_end_ && !m_filter_(*m_begin_))
        {
            ++(m_begin_);
        }
    }

    bool operator==(this_type const &other) const { return m_begin_ == other.m_begin_; }
    bool operator!=(this_type const &other) const { return m_begin_ != other.m_begin_; }

    template <typename Other>
    bool operator==(Other const &other) const { return m_begin_ == other; }

    template <typename Other>
    bool operator!=(Other const &other) const { return m_begin_ != other; }

    reference operator*() const { return *m_begin_; }
    pointer operator->() const { return m_begin_; }

    this_type operator++(int)
    {
        this_type res(*this);
        next();
        return res;
    }

    this_type &operator++()
    {
        next();
        return *this;
    }
    void next()
    {
        ++m_begin_;
        while (m_begin_ != m_end_ && !m_filter_(*m_begin_))
        {
            ++m_begin_;
        }
    }

private:
    filter_type m_filter_;
    base_iterator_type m_begin_, m_end_;
};

template <typename _T1, typename _T2 = _T1>
class SpRange;

template <typename _T1, typename _T2>
SpRange<_T1, _T2> make_range(_T1 const &f, _T2 const &s)
{
    return SpRange<_T1, _T2>(f, s);
}
template <typename _T1, typename _T2>
class SpRange : public std::pair<_T1, _T2>
{
public:
    typedef std::pair<_T1, _T2> base_type;

    typedef SpRange<_T1, _T2> this_type;

    using base_type::first;
    using base_type::second;

    SpRange(){};
    ~SpRange(){};
    SpRange(_T1 const &b, _T2 const &e) : base_type(b, e){};
    SpRange(this_type const &other) : base_type(other){};
    SpRange(this_type &&other) : base_type(std::forward<this_type>(other)){};
    SpRange &operator=(this_type const &) = default;

    bool empty() const { return first == second || same_as(first, second); }
    size_t size() const { return distance(first, second); }

    auto & begin() const { return base_type::first; };
    auto & end() const { return base_type::second; };

    template <typename Pred>
    auto filter(Pred const &pred) const
    {
        return make_range(filtered_iterator(first, second, pred), second);
    }
};

// template <typename _TIterator, typename _Pred>
// class SpIteratorFilter : public _TIterator
// {
//     typedef SpIteratorFilter<_TIterator, _Pred> this_type;
//     typedef _TIterator base_type;
//     typedef _Pred pred_type;
//     using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;

//     SpIteratorFilter(base_type &&self, pred_type &&pred) : base_type(std::forward<base_type>(self)),
//                                                            m_pred_(std::forward<pred_type>(pred)){};
//     ~SpIterator() = default;
//     SpIterator(const this_type &other) : base_type(ohter), m_pred_(other.m_pred_) {}
//     SpIterator(this_type &&other) : m_self_(ohter.m_self_) { other.m_self_ = nullptr; }

//     void next()
//     {
//     }

// private:
//     pred_type m_pred_;
// };

// template <typename... Args, typename _Pred>
// auto filter(SpIterator<Args...> &&it, _Pred &&pred) -> SpIteratorFilter<SpIterator<Args...>, _Pred>
// {
//     return SpIteratorFilter<SpIterator<Args...> >( (std::forward<SpIterator<Args...> >(it), std::forward<_Pred>(pred));
// }

// template <typename _Tp, typename _Pred>
// auto filter(_Tp &&p, _Pred &&pred)
// {
//     return filter(iterator(std::forward<_Tp>(p)), std::forward<_Pred>(pred));
// }

#endif //SP_RANGE_H_