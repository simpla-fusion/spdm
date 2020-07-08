#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include <functional>
#include <iterator>
namespace sp
{

    //##############################################################################################################
    template <typename U>
    class Iterator : public std::iterator<std::input_iterator_tag, U>
    {
    public:
        typedef std::iterator<std::input_iterator_tag, U> base_type;

        typedef Iterator<U> iterator;

        using typename base_type::pointer;
        using typename base_type::reference;
        using typename base_type::value_type;

        typedef std::function<value_type(value_type const &)> next_function_type;

        Iterator() {}
        Iterator(value_type first, next_function_type next_fun) : m_self_(first), m_next_(next_fun) { ; }
        Iterator(iterator const &other) : m_self_(other.m_self_), m_next_(other.m_next_) {}
        Iterator(iterator &&other) : m_self_(other.m_self_), m_next_(std::move(other.m_next_)) { other.m_self_ = nullptr; }

        ~Iterator() {}

        void swap(iterator &other)
        {
            std::swap(m_self_, other.m_self_);
            std::swap(m_next_, other.m_next_);
        }

        iterator &operator=(iterator const &other)
        {
            iterator(other).swap(*this);
            return *this;
        }

        bool operator==(iterator const &other) const { return m_self_ == other.m_self_; }
        bool operator!=(iterator const &other) const { return m_self_ != other.m_self_; }

        iterator &operator++()
        {
            m_self_ = m_next_(m_self_);

            return *this;
        }
        iterator operator++(int)
        {
            iterator res(*this);
            m_self_ = m_next_(m_self_);
            return res;
        }
        reference operator*() { return *m_self_; }
        pointer operator->() { return m_self_.get(); }

    private:
        value_type m_self_;
        next_function_type m_next_;
    };

    //##############################################################################################################
    template <typename T0, typename T1 = T0>
    class Range : public std::pair<T0, T1>
    {

    public:
        typedef std::pair<T0, T1> base_type;

        using base_type::first;
        using base_type::second;

        Range() {}

        template <typename U0, typename U1>
        Range(U0 const &first, U1 const &second) : base_type(T0(first), T2(second)) {}

        template <typename U0, typename U1>
        Range(std::pair<U0, U1> const &p) : base_type(T0(p.first), T2(p.second)) {}

        // virtual ~range(){};

        ptrdiff_t size() { return std::distance(first, second); };

        T0 begin() const { return first; };
        T1 end() const { return second; }
    };

    //##############################################################################################################

} // namespace sp
#endif // SP_RANGE_H_