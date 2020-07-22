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
template <typename... T>
class Iterator;

template <typename T>
struct IteratorTraits : public std::iterator<std::input_iterator_tag, T>
{
    typedef std::iterator<std::input_iterator_tag, T> base_type;
    using base_type::difference_type;
    using base_type::pointer;
    using base_type::reference;
    using base_type ::value_type;
};
template <typename T>
struct IteratorTraits<std::shared_ptr<T>>
{
    typedef ptrdiff_t difference_type;
    typedef std::shared_ptr<T> pointer;
    typedef T& reference;
    typedef T value_type;
};

template <typename T, typename... Others>
struct IteratorProxy;

template <typename T>
class IteratorProxy<T> : public std::iterator<std::input_iterator_tag, T>
{
public:
    typedef IteratorProxy<T> this_type;
    typedef std::iterator<std::input_iterator_tag, T> base_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    IteratorProxy() = default;

    IteratorProxy(const this_type& other) = default;

    IteratorProxy(this_type&& other) = default;

    virtual ~IteratorProxy() = default;

    virtual bool is_derived_from(const std::type_info& tinfo) const { return tinfo == typeid(this_type); }

    virtual std::unique_ptr<this_type> copy() const = 0;

    virtual bool equal(this_type const& other) const { return get_pointer() == other.get_pointer(); };

    virtual bool not_equal(this_type const& other) const { return !equal(other); };

    virtual pointer get_pointer() const = 0;

    virtual reference get_reference() const { return *get_pointer(); }

    virtual void next() = 0;
};

template <typename T, typename V>
class IteratorProxy<T, V,
                    std::enable_if_t<
                        std::is_same_v<V, T*> ||
                            std::is_same_v<V, std::remove_const_t<T>*> ||
                            std::is_same_v<V, std::shared_ptr<T>> ||
                            std::is_same_v<V, std::shared_ptr<std::remove_const_t<T>>>,
                        void>> : public IteratorProxy<T>
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

    IteratorProxy(const this_type& other) : base_type(other), m_base_(other.m_base_), m_pos_(other.m_pos_) {}

    IteratorProxy(this_type&& other) : base_type(other), m_base_(std::move(other.m_base_)), m_pos_(other.m_pos_) {}

    virtual ~IteratorProxy() = default;

    bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); };

    pointer get_pointer() const override { return &(*m_base_) + m_pos_; }

    void next() override { ++m_pos_; }

protected:
    iterator m_base_;
    difference_type m_pos_;
};

template <typename T, typename V>
class IteratorProxy<T, V,
                    std::enable_if_t<
                        std::is_same_v<V&, decltype(std::declval<V>().operator++())> &&
                            std::is_same_v<T&, decltype(std::declval<V>().operator*())>,
                        void>> : public IteratorProxy<T>
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

    bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); };

    pointer get_pointer() const { return &get_reference(); }

    reference get_reference() const { return (*m_base_); }

    void next() override { ++m_base_; }

protected:
    iterator m_base_;
};
template <typename T, typename V, typename Mapper>
struct IteratorProxy<T, Iterator<V>, Mapper> : public IteratorProxy<T>
{
public:
    typedef Mapper mapper_t;
    typedef IteratorProxy<T, V, mapper_t> this_type;
    typedef IteratorProxy<T> base_type;
    typedef V iterator;

    using typename base_type::pointer;
    using typename base_type::value_type;

    IteratorProxy(iterator const& it, mapper_t const& mapper) : m_it_(it), m_mapper_(mapper) {}
    IteratorProxy(iterator&& it, mapper_t&& mapper) : m_it_(std::move(it)), m_mapper_(std::move(mapper)) {}
    IteratorProxy(this_type const& other) : m_it_(other.m_it_), m_mapper_(other.m_mapper_) {}
    IteratorProxy(this_type&& other) : m_it_(std::move(other.m_it_)), m_mapper_(std::move(other.m_mapper_)) {}
    ~IteratorProxy() = default;

    bool is_derived_from(const std::type_info& tinfo) const override { return tinfo == typeid(this_type) || base_type::is_derived_from(tinfo); }

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() const { return m_mapper_(*m_it_); }

    void next() { ++m_it_; }

private:
    iterator m_it_;
    mapper_t m_mapper_;
};

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
class Iterator<T> : public IteratorTraits<T>
{
public:
    typedef IteratorTraits<T> base_type;

    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    template <typename... U>
    friend class Iterator;

    Iterator() : m_proxy_(nullptr) {}

    Iterator(nullptr_t) = delete;

    explicit Iterator(pointer p) : m_proxy_(nullptr) {}

    template <typename... Args>
    Iterator(Args&&... args) : m_proxy_(make_iterator_proxy(std::forward<Args>(args)...)) {}

    Iterator(Iterator const& other) : m_proxy_(other.m_proxy_ == nullptr ? nullptr : other.m_proxy_->copy()) {}

    Iterator(Iterator&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Iterator() {}

    void swap(Iterator& other)
    {
        std::swap(m_proxy_, other.m_proxy_);
    }

    Iterator& operator=(Iterator const& other)
    {
        Iterator(other).swap(*this);
        return *this;
    }

    bool operator==(Iterator const& other) const { return m_proxy_->equal(*other.m_proxy_); }

    bool operator!=(Iterator const& other) const { return m_proxy_->not_equal(*other.m_proxy_); }

    bool operator==(pointer other) const { return m_proxy_->get_pointer() = other; }

    bool operator!=(pointer other) const { return m_proxy_->get_pointer() != other; }

    operator bool() const { return m_proxy_->get_pointer() != nullptr; }

    Iterator&
    operator++()
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

    reference operator*() { return m_proxy_->get_reference(); }

    pointer operator->() { return m_proxy_->get_pointer(); }

private:
    std::unique_ptr<IteratorProxy<T>> m_proxy_;

    auto make_iterator_proxy()
    {
        return new IteratorProxy<value_type>();
    }
    auto make_iterator_proxy(value_type* p)
    {
        return new IteratorProxy<value_type>(p);
    }

    template <typename TI>
    auto make_iterator_proxy(const TI& it)
    {
        return new IteratorProxy<value_type, TI>(it);
    }
    template <typename TI, typename Mapper>
    auto make_iterator_proxy(const TI& it, const Mapper& mapper)
    {
        return new IteratorProxy<value_type, TI, Mapper>(it, mapper);
    }

    template <typename TI>
    auto make_iterator_proxy(const TI& it, const std::function<bool(const value_type&)>& filter)
    {
        return new IteratorProxy<value_type, TI, std::function<bool(const value_type&)>>(it, filter);
    }

    template <typename V>
    auto make_iterator_proxy(const Iterator<V>& other, std::enable_if_t<std::is_same_v<value_type, const V>, void*> _ = nullptr)
    {
        return other.m_proxy_->const_copy();
    }
    template <typename V>
    auto make_iterator_proxy(const Iterator<V>& other, std::enable_if_t<std::is_same_v<value_type, V>, void*> _ = nullptr)
    {
        return other.m_proxy_->copy();
    }
};

template <typename... T>
class Iterator : public Iterator<std::tuple<T...>>
{
    typedef Iterator<std::tuple<T...>> base_type;
    using base_type::Iterator;
};

} // namespace sp
#endif //SP_ITERATOR_H_