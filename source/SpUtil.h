#ifndef SP_UTIL_H_
#define SP_UTIL_H_
#include <boost/format.hpp>
#include <exception>
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

    class NotImplementedException : public std::logic_error
    {
    public:
        NotImplementedException(std::string const &prefix = "") : std::logic_error{prefix + " Function  not yet implemented."} {}
    };
} // namespace sp

#if __GNUC__
#define NOT_IMPLEMENTED                                                                                                     \
    {                                                                                                                       \
        throw sp::NotImplementedException((boost::format("[%s:%d][%s]") % __FILE__ % __LINE__ % __PRETTY_FUNCTION__).str()); \
    }
#else
#define NOT_IMPLEMENTED                                                             \
    {                                                                               \
        throw sp::NotImplementedException("[" __FILE__ ":" __LINE__ "]:" __FUNC__); \
    }
#endif
#endif //SP_UTIL_H_