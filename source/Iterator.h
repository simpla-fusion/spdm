#ifndef SP_ITERATOR_H_
#define SP_ITERATOR_H_

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
class Iterator;

template <typename T>
class IteratorProxy<T>
{
public:
    typedef IteratorProxy<T> this_type;
    typedef std::iterator_traits<T*> traits_type;
    typedef typename traits_type::pointer pointer;
    typedef typename traits_type::reference reference;
    typedef typename traits_type::value_type value_type;

    IteratorProxy() {}

    virtual ~IteratorProxy() {}

    virtual bool is_derived_from(const std::type_info& tinfo) const { return tinfo == typeid(this_type); }

    virtual bool equal(this_type const& other) const { return false; }

    virtual this_type* copy() const { return nullptr; }

    virtual IteratorProxy<const T>* const_copy() const { return nullptr; }

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

    IteratorProxy(iterator&& it) : m_it_(std::move(it)) {}
    IteratorProxy(const iterator& it) : m_it_(it) {}
    IteratorProxy(const this_type& other) : m_it_(other.m_it_) {}
    IteratorProxy(this_type&& other) : m_it_(std::move(other.m_it_)) {}
    virtual ~IteratorProxy() {}

    bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

    base_type* copy() const override { return new this_type(*this); };

    IteratorProxy<const T>* const_copy() const { return new IteratorProxy<const T, IT>(m_it_); }

    bool equal(base_type const& other) const override
    {

        return other.is_derived_from(typeid(this_type)) && dynamic_cast<this_type const&>(other).m_it_ == m_it_;
    }

    void next() override { ++m_it_; }

    // pointer get() const { return m_it_.operator->(); }

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

    IteratorProxy(iterator const& it, mapper_t const& mapper) : base_type(it), m_mapper_(mapper) {}
    IteratorProxy(iterator&& it, mapper_t&& mapper) : base_type(std::move(it)), m_mapper_(std::move(mapper)) {}
    IteratorProxy(this_type const& other) : base_type(other.m_it_), m_mapper_(other.m_mapper_) {}
    IteratorProxy(this_type&& other) : base_type(std::forward<this_type>(other)), m_mapper_(std::move(other.m_mapper_)) {}
    ~IteratorProxy() {}

    bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

    base_type* copy() const override { return new this_type(*this); }
    IteratorProxy<const T>* const_copy() const { return new IteratorProxy<const T, IT, Mapper>(m_it_, m_mapper_); }

    pointer get() const override { return m_mapper_(m_it_); }

private:
    mapper_t m_mapper_;
};

// template <typename T, typename... Args>
// IteratorProxy<const T, std::remove_const_t<std::remove_reference_t<Args>>...> *
// make_const_iterator_proxy(Args &&... args)
// {
//     return new IteratorProxy<const T, std::remove_const_t<std::remove_reference_t<Args>>...>(std::forward<Args>(args)...);
// }
// template <typename T, typename... Args>
// IteratorProxy<const T, Args...> *
// make_const_iterator_proxy(IteratorProxy<T, Args...> const &other)
// {
//     return other.as_const_copy();
// }
// template <typename T, typename... Args>
// auto make_iterator_proxy(Args &&... args) -> std::enable_if<std::is_const_v<T>,
//                                                             decltype(make_const_iterator_proxy<std::remove_const_t<T>>(std::forward<Args>(args)...))>
// {
//     return make_const_iterator_proxy<std::remove_const_t<T>>(std::forward<Args>(args)...);
// }

template <typename T>
class Iterator : public std::iterator_traits<T*>
{
public:
    typedef std::iterator_traits<T*> traits_type;

    using typename traits_type::pointer;
    using typename traits_type::reference;
    using typename traits_type::value_type;

    template <typename U>
    friend class Iterator;

    Iterator() {}

    // template <typename... Args>
    // Iterator(Args&&... args) : m_proxy_(make_iterator_proxy<T>(std::forward<Args>(args)...)) {}

    Iterator(Iterator const& other) : m_proxy_(other.m_proxy_->copy()) {}

    Iterator(Iterator&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Iterator() {}

    void swap(Iterator& other) { std::swap(m_proxy_, other.m_proxy_); }

    Iterator& operator=(Iterator const& other)
    {
        Iterator(other).swap(*this);
        return *this;
    }

    bool operator==(Iterator const& other) const { return m_proxy_->equal(*other.m_proxy_); }

    bool operator!=(Iterator const& other) const { return !m_proxy_->equal(*other.m_proxy_); }

    Iterator& operator++()
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

    template <typename U>
    auto make_iterator_proxy()
    {
        return new IteratorProxy<U>();
    }
    template <typename U, typename TI>
    auto make_iterator_proxy(TI const& it)
    {
        return new IteratorProxy<U, TI>(it);
    }
    template <typename U, typename TI, typename Mapper>
    auto make_iterator_proxy(TI const& it, Mapper const& mapper)
    {
        return new IteratorProxy<U, TI, Mapper>(it, mapper);
    }
    template <typename U, typename V>
    auto make_iterator_proxy(Iterator<V> const& other, std::enable_if_t<std::is_same_v<U, const V>, void*> _ = nullptr)
    {
        return other.m_proxy_->const_copy();
    }
    template <typename U, typename V>
    auto make_iterator_proxy(Iterator<V> const& other, std::enable_if_t<std::is_same_v<U, V>, void*> _ = nullptr)
    {
        return other.m_proxy_->copy();
    }
};

} // namespace sp
#endif //SP_ITERATOR_H_