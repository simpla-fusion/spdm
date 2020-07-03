#ifndef SP_UTIL_H_
#define SP_UTIL_H_
#include <functional>
#include <string>
#include <tuple>

namespace sp
{

    std::string urljoin(std::string const &base, std::string const &path);

    /***
 * Return:   
 *    tuple( scheme ,authority , path,query ,fragment )
 */
    std::tuple<std::string, std::string, std::string, std::string, std::string> urlparser(std::string const &url);

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

    template <typename T, typename... Others>
    class Iterator
    {
    public:
        typedef Iterator<T, Others...> this_type;
        typedef T value_type;
        typedef value_type *pointer;
        typedef value_type &reference;

        Iterator(pointer d = nullptr) : m_self_(d){};
        ~Iterator() = default;
        Iterator(this_type const &) = default;
        Iterator(this_type &&) = default;

        this_type &operator=(this_type const &) = default;

        bool operator==(this_type const &other) const { return m_self_ == other.m_self_  || same_as(*m_self_, *other.m_self_); }
        bool operator!=(this_type const &other) const { return !(*this == other); }
        ptrdiff_t operator-(this_type const &other) const { return distance(*this, other); }

        reference operator*() const { return *m_self_; };
        pointer operator->() const { return m_self_; };

        this_type operator++(int)
        {
            this_type res(*this);
            m_self_ = next(m_self_);
            return res;
        }

        this_type &operator++()
        {
            m_self_ = next(m_self_);
            return *this;
        }

    private:
        pointer m_self_;
    };

    template <typename T>
    struct iterator_trait
    {
        typedef Iterator<T> type;
    };

    template <typename T, typename... Others>
    struct iterator_trait<Iterator<T, Others...>>
    {
        typedef Iterator<T, Others...> type;
    };
    template <typename T>
    using iterator_trait_t = typename iterator_trait<T>::type;

    template <typename T1, typename T2 = T1>
    class Range : public std::pair<iterator_trait_t<T1>, iterator_trait_t<T2>>
    {

    public:
        typedef std::pair<iterator_trait_t<T1>, iterator_trait_t<T2>> base_type;
        typedef Range<T1, T2> this_type;
        typedef iterator_trait_t<T1> iterator;
        typedef iterator_trait_t<T2> iterator_end;
        using base_type::first;
        using base_type::second;

        Range(T1 const &a, T2 const &b) : base_type(iterator(a), iterator(b)) {}
        Range() = default;
        ~Range() = default;
        Range(Range const &) = default;
        Range(Range &&) = default;

        size_t size() const { return std::distance(first, second); }
        size_t empty() const { return first == second; }

        auto begin() const { return base_type::first; }
        auto end() const { return base_type::first; }
    };

} // namespace sp

#endif //SP_UTIL_H_