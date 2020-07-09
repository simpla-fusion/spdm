#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include <functional>
#include <iterator>
#include <memory>
namespace sp
{

    //##############################################################################################################
    // iterator
    template <typename T, typename... Others>
    struct IteratorProxy;

    template <typename T>
    class IteratorProxy<T>
    {
    public:
        typedef IteratorProxy<T> this_type;
        typedef std::iterator_traits<T> traits_type;
        typedef typename traits_type::pointer pointer;
        typedef typename traits_type::reference reference;
        typedef typename traits_type::value_type value_type;

        IteratorProxy() {}

        virtual ~IteratorProxy() {}

        bool equal(this_type const &other) const { return false; }

        virtual this_type *copy() const { return nullptr; }

        virtual pointer next() { return nullptr; }
    };

    template <typename T, typename IT>
    class IteratorProxy<T, IT> : public IteratorProxy<T>
    {
    public:
        typedef IteratorProxy<T> base_type;
        using typename base_type::pointer;
        using typename base_type::reference;
        using typename base_type::value_type;

        typedef IteratorProxy<T, IT> this_type;

        typedef IT iterator;

        IteratorProxy() {}
        IteratorProxy(iterator const &it) : m_it_(it) {}
        IteratorProxy(this_type const &other) : m_it_(other.m_it_) {}
        IteratorProxy(this_type &&other) : m_it_(std::move(other.m_it_)) {}

        virtual ~IteratorProxy() {}

        bool equal(base_type const &other) const { return dynamic_cast<this_type const &>(other).m_it_ == m_it_; }

        virtual base_type *copy() const { return new this_type(*this); };

        virtual pointer next() { return pointer(); };

    protected:
        iterator m_it_;
    };

    template <typename T, typename IT, typename Mapper>
    struct IteratorProxy<T, IT, Mapper> : public IteratorProxy<T, IT>
    {
    public:
        typedef IteratorProxy<T, IT, Mapper> this_type;
        typedef IteratorProxy<T, IT> base_type;
        typedef IT iterator;
        using base_type::m_it_;
        using typename base_type::pointer;

        typedef std::function<pointer(iterator const &)> mapper_t;

        IteratorProxy(iterator const &it) : base_type(it) {}
        IteratorProxy(iterator const &it, mapper_t const &mapper) : base_type(it), m_mapper_(mapper) {}
        IteratorProxy(this_type const &other) : base_type(other.m_it_), m_mapper_(other.m_mapper_) {}
        IteratorProxy(this_type &&other) : base_type(std::forward<this_type>(other)), m_mapper_(std::move(other.m_mapper_)) {}
        ~IteratorProxy() {}

        base_type *copy() const { return new this_type(*this); }

        pointer next()
        {
            pointer p = m_mapper_(m_it_);
            ++m_it_;
            return p;
        }

    private:
        mapper_t m_mapper_;
    };

    template <typename T, typename... Args>
    IteratorProxy<T, std::remove_const_t<std::remove_reference_t<Args>>...> *make_iterator_proxy(Args &&... args)
    {
        return new IteratorProxy<T, std::remove_const_t<std::remove_reference_t<Args>>...>(std::forward<Args>(args)...);
    }

    template <typename T>
    class Iterator : public std::iterator_traits<T>
    {
    public:
        typedef std::iterator_traits<T> traits_type;

        using typename traits_type::pointer;
        using typename traits_type::reference;
        using typename traits_type::value_type;

        Iterator() {}

        template <typename... Args>
        Iterator(Args &&... args) : m_proxy_(make_iterator_proxy<T>(std::forward<Args>(args)...)),
                                    m_current_(m_proxy_->next()) {}

        Iterator(Iterator const &other) : m_proxy_(other.m_proxy_->copy()), m_current_(other.m_current_) {}

        Iterator(Iterator &&other) : m_proxy_(other.m_proxy_.release()), m_current_(other.m_current_) {}

        ~Iterator() {}

        void swap(Iterator &other)
        {
            std::swap(m_current_, other.m_current_);
            std::swap(m_proxy_, other.m_proxy_);
        }

        Iterator &operator=(Iterator const &other)
        {
            Iterator(other).swap(*this);
            return *this;
        }

        bool operator==(Iterator const &other) const { return m_proxy_->equal(*other.m_proxy_); }

        bool operator!=(Iterator const &other) const { return !m_proxy_->equal(*other.m_proxy_); }

        Iterator &operator++()
        {
            m_current_ = m_proxy_->next();
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator res(*this);
            m_current_ = m_proxy_.next();
            return res;
        }

        reference operator*() { return *m_current_; }

        pointer operator->() { return m_current_; }

    private:
        pointer m_current_;

        std::unique_ptr<IteratorProxy<T>> m_proxy_;
    };

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

        Range(iterator const &first, iterator const &second) : base_type(first, second) {}

        template <typename U0, typename U1>
        Range(U0 const &first, U1 const &second) : Range(iterator(first), iterator(second)) {}

        template <typename U0, typename U1>
        Range(std::tuple<U0, U1> const &r) : Range(std::get<0>(r), std::get<1>(r)) {}

        Range(base_type const &p) : base_type(p) {}

        ~Range(){};

        iterator begin() const { return first; }

        iterator end() const { return second; }
    };
    //##############################################################################################################

} // namespace sp
#endif // SP_RANGE_H_