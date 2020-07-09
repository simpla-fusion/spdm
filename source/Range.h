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

        virtual bool is_derived_from(const std::type_info &tinfo) const { return tinfo == typeid(this_type); }

        virtual bool equal(this_type const &other) const { return false; }

        virtual this_type *copy() const { return nullptr; }

        virtual void next() {}

        virtual pointer get() const { return nullptr; }
    };

    template <typename T, typename IT>
    class IteratorProxy<T, IT> : public IteratorProxy<T>
    {
    public:
        typedef IteratorProxy<T> base_type;
        typedef IteratorProxy<T, IT> this_type;
        using typename base_type::pointer;
        using typename base_type::reference;
        using typename base_type::value_type;

        typedef IT iterator;

        IteratorProxy(iterator const &it) : m_it_(it) {}
        IteratorProxy(this_type const &other) : m_it_(other.m_it_) {}
        IteratorProxy(this_type &&other) : m_it_(std::move(other.m_it_)) {}
        virtual ~IteratorProxy() {}

        bool is_derived_from(const std::type_info &tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

        base_type *copy() const override { return new this_type(*this); };

        bool equal(base_type const &other) const override
        {

            return other.is_derived_from(typeid(this_type)) && dynamic_cast<this_type const &>(other).m_it_ == m_it_;
        }

        void next() override { ++m_it_; }

    protected:
        iterator m_it_;
    };

    template <typename T, typename IT, typename Mapper>
    struct IteratorProxy<T, IT, Mapper> : public IteratorProxy<T, IT>
    {
    public:
        typedef IteratorProxy<T, IT, Mapper> this_type;
        typedef IteratorProxy<T, IT> base_type;
        typedef Mapper mapper_t;

        using base_type::m_it_;

        using typename base_type::iterator;
        using typename base_type::pointer;

        IteratorProxy(iterator const &it, mapper_t const &mapper) : base_type(it), m_mapper_(mapper) {}
        IteratorProxy(this_type const &other) : base_type(other.m_it_), m_mapper_(other.m_mapper_) {}
        IteratorProxy(this_type &&other) : base_type(std::forward<this_type>(other)), m_mapper_(std::move(other.m_mapper_)) {}
        ~IteratorProxy() {}

        bool is_derived_from(const std::type_info &tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

        base_type *copy() const override { return new this_type(*this); }

        pointer get() const override { return m_mapper_(m_it_); }

    private:
        mapper_t m_mapper_;
    };

    template <typename T, typename... Args>
    IteratorProxy<T, std::remove_const_t<std::remove_reference_t<Args>>...> *
    make_iterator_proxy(Args &&... args)
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
        Iterator(Args &&... args) : m_proxy_(make_iterator_proxy<T>(std::forward<Args>(args)...)) {}

        Iterator(Iterator const &other) : m_proxy_(other.m_proxy_->copy()) {}

        Iterator(Iterator &&other) : m_proxy_(other.m_proxy_.release()) {}

        ~Iterator() {}

        void swap(Iterator &other) { std::swap(m_proxy_, other.m_proxy_); }

        Iterator &operator=(Iterator const &other)
        {
            Iterator(other).swap(*this);
            return *this;
        }

        bool operator==(Iterator const &other) const { return m_proxy_->equal(*other.m_proxy_); }

        bool operator!=(Iterator const &other) const { return !m_proxy_->equal(*other.m_proxy_); }

        Iterator &operator++()
        {
            m_proxy_->next();
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator res(*this);
            m_proxy_->next();
            return res;
        }

        reference operator*() { return *m_proxy_->get(); }

        pointer operator->() { return m_proxy_->get(); }

    private:
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