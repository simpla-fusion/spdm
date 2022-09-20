#ifndef SP_ITERATOR_H_
#define SP_ITERATOR_H_

#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
namespace sp
{

//##############################################################################################################
// iterator

template <typename P, typename Q = P, typename Enable = void>
struct IteratorProxy;

template <typename T>
class IteratorProxy
{
public:
    typedef IteratorProxy<T> this_type;

    typedef std::input_iterator_tag iterator_category;
    typedef T value_type;
    typedef T reference_type;
    typedef T* pointer_type;

    IteratorProxy() = default;

    IteratorProxy(const this_type& other) = default;

    IteratorProxy(this_type&& other) = default;

    virtual ~IteratorProxy() = default;

    virtual std::unique_ptr<this_type> copy() const { return std::make_unique<this_type>(); };

    virtual bool equal(this_type const& other) const { return get_pointer() == other.get_pointer(); };

    virtual bool not_equal(this_type const& other) const { return !equal(other); };

    virtual pointer get_pointer() const = delete;

    virtual reference get_reference() const { return T(); }

    virtual void next(){};
};
template <typename T>
class IteratorProxy<T&>
{
public:
    typedef IteratorProxy<T&> this_type;

    typedef std::output_iterator_tag iterator_category;
    typedef T value_type;
    typedef T& reference_type;
    typedef T* pointer_type;

    IteratorProxy() = default;

    IteratorProxy(const this_type& other) = default;

    IteratorProxy(this_type&& other) = default;

    virtual ~IteratorProxy() = default;

    virtual std::unique_ptr<this_type> copy() const { return std::make_unique<this_type>(); };

    virtual bool equal(this_type const& other) const { return get_pointer() == other.get_pointer(); };

    virtual bool not_equal(this_type const& other) const { return !equal(other); };

    virtual pointer get_pointer() const { return get_reference(); };

    virtual reference get_reference() const = 0;

    virtual void next(){};
};

template <typename T>
class IteratorProxy<T*>
{
public:
    typedef IteratorProxy<T*> this_type;

    typedef std::output_iterator_tag iterator_category;
    typedef T value_type;
    typedef T& reference_type;
    typedef T* pointer_type;

    IteratorProxy() = default;

    IteratorProxy(const this_type& other) = default;

    IteratorProxy(this_type&& other) = default;

    virtual ~IteratorProxy() = default;

    virtual std::unique_ptr<this_type> copy() const { return std::make_unique<this_type>(); };

    virtual bool equal(this_type const& other) const { return get_pointer() == other.get_pointer(); };

    virtual bool not_equal(this_type const& other) const { return !equal(other); };

    virtual pointer get_pointer() const { return nullptr; };

    virtual reference get_reference() const { return *get_pointer(); };

    virtual void next(){};
};

template <typename T, typename V>
class IteratorProxy<T, V,
                    std::enable_if_t<                                              //
                        std::is_same_v<V, std::shared_ptr<T>> ||                   //
                        std::is_same_v<V, std::shared_ptr<std::remove_const_t<T>>> //
                        >> : public IteratorProxy<T>
{

public:
    typedef IteratorProxy<T> base_type;
    typedef IteratorProxy<T, V> this_type;
    typedef V iterator;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    IteratorProxy(iterator&& it, difference_type pos = 0) : m_base_(std::move(it)), m_pos_(pos) {}

    IteratorProxy(const iterator& it, difference_type pos = 0) : m_base_(it), m_pos_(pos) {}

    IteratorProxy(const this_type& other) : m_base_(other.m_base_), m_pos_(other.m_pos_) {}

    IteratorProxy(this_type&& other) : m_base_(std::move(other.m_base_)), m_pos_(other.m_pos_) {}

    virtual ~IteratorProxy() = default;

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); };

    pointer get_pointer() const override { return &(*m_base_) + m_pos_; }

    void next() override { ++m_pos_; }

protected:
    iterator m_base_;
    difference_type m_pos_;
};

template <typename T, typename V>
class IteratorProxy<T, V, std::enable_if_t<std::is_same_v<T, typename std::iterator_traits<V>::value_type>>> : public IteratorProxy<T>
{
public:
    typedef IteratorProxy<T> base_type;
    typedef IteratorProxy<T, V> this_type;
    typedef V iterator;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    IteratorProxy(iterator&& it, difference_type pos = 0) : m_base_(std::move(it)) {}

    IteratorProxy(const iterator& it, difference_type pos = 0) : m_base_(it) {}

    IteratorProxy(const this_type& other) : base_type(other), m_base_(other.m_base_) {}

    IteratorProxy(this_type&& other) : base_type(other), m_base_(std::move(other.m_base_)) {}

    virtual ~IteratorProxy() = default;

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); };

    pointer get_pointer() const override { return &get_reference(); }

    reference get_reference() const override { return (*m_base_); }

    void next() override { ++m_base_; }

protected:
    iterator m_base_;
};

template <typename T>
class IteratorProxy<T, IteratorProxy<T>> : public IteratorProxy<T>
{
public:
    typedef IteratorProxy<T> base_type;
    typedef IteratorProxy<T, IteratorProxy<T>> this_type;
    typedef IteratorProxy<T> iterator;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    IteratorProxy(IteratorProxy<T>* base) : m_base_(base) {}

    IteratorProxy(const this_type& other) : m_base_(other.m_base_) {}

    IteratorProxy(this_type&& other) : m_base_(std::move(other.m_base_)) {}

    virtual ~IteratorProxy() = default;

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); };

    pointer get_pointer() const override { return m_base_->get_pointer(); }

    reference get_reference() const override { return m_base_->get_reference(); }

    void next() override { m_base_->next(); }

protected:
    std::unique_ptr<IteratorProxy<T>> m_base_;
};

template <typename T, typename V>
class IteratorProxy<T, IteratorProxy<V>> : public IteratorProxy<T>
{
public:
    typedef IteratorProxy<T> base_type;
    typedef IteratorProxy<T, IteratorProxy<V>> this_type;
    typedef IteratorProxy<T> iterator;

    typedef std::function<T(const V&)> mapper_t;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    IteratorProxy(IteratorProxy<T>* base, const mapper_t mapper) : m_base_(base), m_mapper_(mapper) {}

    IteratorProxy(const this_type& other) : m_base_(other.m_base_), m_mapper_(other.m_mapper_) {}

    IteratorProxy(this_type&& other) : m_base_(std::move(other.m_base_)), m_mapper_(other.m_mapper_) {}

    virtual ~IteratorProxy() = default;

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); };

    pointer get_pointer() const override = delete;

    reference get_reference() const override { return m_mapper_(m_base_->get_reference()); }

    void next() override { m_base_->next(); }

protected:
    std::unique_ptr<IteratorProxy<T>> m_base_;
    mapper_t m_mapper_;
};

// template <typename T, typename V, typename Mapper>
// struct IteratorProxy<T, Iterator<V>, Mapper> : public IteratorProxy<T>
// {
// public:
//     typedef Mapper mapper_t;
//     typedef IteratorProxy<T, V, mapper_t> this_type;
//     typedef IteratorProxy<T> base_type;
//     typedef V iterator;

//     using typename base_type::pointer;
//     using typename base_type::value_type;

//     IteratorProxy(iterator const& it, mapper_t const& mapper) : m_it_(it), m_mapper_(mapper) {}
//     IteratorProxy(iterator&& it, mapper_t&& mapper) : m_it_(std::move(it)), m_mapper_(std::move(mapper)) {}
//     IteratorProxy(this_type const& other) : m_it_(other.m_it_), m_mapper_(other.m_mapper_) {}
//     IteratorProxy(this_type&& other) : m_it_(std::move(other.m_it_)), m_mapper_(std::move(other.m_mapper_)) {}
//     ~IteratorProxy() = default;

//     bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

//     std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); }

//     pointer get_pointer() const { return m_mapper_(*m_it_); }

//     void next() { ++m_it_; }

// private:
//     iterator m_it_;
//     mapper_t m_mapper_;
// };

// template <typename T>
// class IteratorProxy<T, std::shared_ptr<std::remove_const_t<T>>> : public IteratorProxy<T>
// {
// public:
//     typedef IteratorProxy<T> base_type;
//     typedef IteratorProxy<T, std::shared_ptr<std::remove_const_t<T>>> this_type;

//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;

//     typedef std::shared_ptr<std::remove_const_t<T>> iterator;

//     IteratorProxy(iterator&& it) : m_it_(std::move(it)), m_pos_(0) {}

//     IteratorProxy(const iterator& it) : m_it_(it), m_pos_(0) {}

//     IteratorProxy(const this_type& other) : base_type(other), m_it_(other.m_it_), m_pos_(other.m_pos_) {}

//     IteratorProxy(this_type&& other) : base_type(other), m_it_(std::move(other.m_it_)), m_pos_(other.m_pos_) {}

//     virtual ~IteratorProxy() = default;

//     bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

//     base_type* copy() const override { return new this_type(*this); };

//     IteratorProxy<const T>* const_copy() const { return new IteratorProxy<const T, iterator>(m_it_); }

//     pointer next() override
//     {
//         ++m_pos_;
//         return m_it_.get() + m_pos_ - 1;
//     }

// protected:
//     iterator m_it_;
//     size_t m_pos_;
// };

// template <typename U, typename... V, typename Mapper>
// struct IteratorProxy<U, Iterator<V...>, Mapper> : public IteratorProxy<U>
// {
// public:
//     typedef Mapper mapper_t;
//     typedef Iterator<V...> iterator;
//     typedef IteratorProxy<U, iterator, mapper_t> this_type;
//     typedef IteratorProxy<U> base_type;

//     using typename base_type::pointer;
//     using typename base_type::value_type;

//     IteratorProxy(const iterator& it, const mapper_t& mapper) : m_it_(it), m_mapper_(mapper) {}
//     IteratorProxy(iterator&& it, mapper_t&& mapper) : m_it_(std::move(it)), m_mapper_(std::move(mapper)) {}
//     IteratorProxy(const this_type& other) : m_it_(other.m_it_), m_mapper_(other.m_mapper_) {}
//     IteratorProxy(this_type&& other) : m_it_(std::move(other.m_it_)), m_mapper_(std::move(other.m_mapper_)) {}
//     ~IteratorProxy() = default;

//     bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

//     base_type* copy() const override { return new this_type(*this); }

//     IteratorProxy<const value_type>* const_copy() const { return new IteratorProxy<const value_type, iterator, mapper_t>(m_it_, m_mapper_); }

//     pointer next() override
//     {
//         pointer p = nullptr;

//         // try
//         // {
//         //     p = m_mapper_(m_it_);
//         //     ++m_it_;
//         // }
//         // catch (...)
//         // {
//         //     p = nullptr;
//         // }

//         return p;
//     }

// private:
//     iterator m_it_;
//     mapper_t m_mapper_;
// };
template <typename T>
std::unique_ptr<IteratorProxy<T>> make_iterator_proxy() { return std::make_unique<IteratorProxy<T>>(); }

template <typename T, typename V, typename... Args>
std::unique_ptr<IteratorProxy<T>> make_iterator_proxy(const V& v, Args&&... args) { return std::make_unique<IteratorProxy<T, V>>(v, std::forward<Args>(args)...); }

template <typename T, typename V, typename... Args>
std::unique_ptr<IteratorProxy<T>> make_iterator_proxy(const Iterator<V>& v, Args&&... args) { return std::make_unique<IteratorProxy<T, IteratorProxy<V>>>(v.m_proxy_->copy(), std::forward<Args>(args)...); }

template <typename T>
class Iterator
{
public:
    typedef Iterator<T> this_type;
    typedef T value_type;

    template <typename... Args>
    Iterator(Args&&... args) : m_proxy_(make_iterator_proxy<value_type>(std::forward<Args>(args)...)) {}

    Iterator(const this_type& other) : m_proxy_(other.m_proxy_ == nullptr ? nullptr : other.m_proxy_->copy()) {}

    Iterator(this_type&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Iterator() {}

    void swap(this_type& other) { std::swap(m_proxy_, other.m_proxy_); }

    this_type& operator=(const this_type& other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    bool operator==(this_type const& other) const { return m_proxy_->equal(*other.m_proxy_); }

    bool operator!=(this_type const& other) const { return m_proxy_->not_equal(*other.m_proxy_); }

    template <typename V>
    bool operator==(const V& other) const { return m_proxy_->equal(other); }

    template <typename V>
    bool operator!=(const V& other) const { return m_proxy_->not_equal(other); }

    operator bool() const { return !m_proxy_->is_null(); }

    this_type& operator++()
    {
        m_proxy_->next();
        return *this;
    }

    this_type operator++(int)
    {
        this_type res(*this);
        m_proxy_->next();
        return res;
    }

    auto operator*() { return m_proxy_->get_reference(); }

    auto operator-> () { return m_proxy_->get_pointer(); }

private:
    std::unique_ptr<IteratorProxy<T>> m_proxy_;
};

} // namespace sp
#endif //SP_ITERATOR_H_