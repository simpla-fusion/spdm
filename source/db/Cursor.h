#ifndef SPDB_Cursor_h_
#define SPDB_Cursor_h_
#include "../utility/Logger.h"
#include "../utility/TypeTraits.h"
#include <functional>
#include <iterator>
#include <memory>
namespace sp
{
namespace db
{
template <typename>
class Cursor;

template <typename U, typename V = U, typename Enable = void>
struct CursorProxy;

template <typename U>
struct CursorProxy<U>
{
    typedef CursorProxy<U> this_type;

    typedef typename std::iterator_traits<U*>::value_type value_type;
    typedef typename std::iterator_traits<U*>::reference reference;
    typedef typename std::iterator_traits<U*>::pointer pointer;
    typedef typename std::iterator_traits<U*>::difference_type difference_type;

    CursorProxy() = default;

    virtual ~CursorProxy() = default;

    virtual std::unique_ptr<this_type> copy() const = 0;

    virtual bool next() = 0;

    virtual bool done() const = 0;

    virtual pointer get_pointer() = 0;

    virtual reference get_reference() { return *get_pointer(); }
};

template <typename U>
struct CursorProxy<U, std::nullptr_t> : public CursorProxy<U>
{
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, std::nullptr_t> this_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy() = default;
    ~CursorProxy() = default;

    std::unique_ptr<base_type> copy() const final { return std::make_unique<this_type>(); }

    bool next() final { return false; };

    bool done() const final { return true; };

    reference get_reference() final { return *get_pointer(); }

    pointer get_pointer() final { return nullptr; }
};

template <typename U>
class CursorProxy<U, std::shared_ptr<U>> : public CursorProxy<U>
{

public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, std::shared_ptr<U>> this_type;
    typedef std::shared_ptr<U> base_iterator;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(const base_iterator& ib, difference_type pos = 0) : m_base_(ib), m_end_(pos), m_pos_(0) {}

    CursorProxy(const base_iterator& ib, const base_iterator& ie) : m_base_(ib), m_end_(std::distance(ib, ie)), m_pos_(0) {}

    ~CursorProxy() = default;

    bool done() const { return m_pos_ >= m_end_; }

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() override { return m_pos_ >= m_end_ ? nullptr : m_base_.get() + m_pos_; }

    bool next() override
    {
        if (m_pos_ < m_end_)
        {
            ++m_pos_;
        }

        return !done();
    }

protected:
    base_iterator m_base_;
    ptrdiff_t m_end_, m_pos_;
};

template <typename U, typename V>
class CursorProxy<U, V, std::enable_if_t<std::is_same<typename std::iterator_traits<V>::pointer, typename CursorProxy<U>::pointer>::value>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, V> this_type;
    typedef V iterator;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(const iterator& ib, const iterator& ie) : m_it_(ib), m_ie_(ie) {}
    CursorProxy(const iterator& ib) : m_it_(ib), m_ie_(ib) { ++m_ie_; }

    ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    bool done() const override { return m_it_ == m_ie_; }

    pointer get_pointer() override { return m_it_.operator->(); }

    bool next() override
    {
        if (m_it_ != m_ie_)
        {
            ++m_it_;
        }
        return !done();
    }

protected:
    iterator m_it_, m_ie_;
};

namespace _detail
{
// filter

template <typename U, typename Fitler>
class CursorFilter : public CursorProxy<U>
{
public:
    typedef U value_type;
    typedef CursorProxy<U> base_type;
    typedef CursorFilter<U, Fitler> this_type;
    typedef Fitler filter_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;

    CursorFilter(CursorProxy<U>* base, const filter_type filter) : m_base_(base), m_filter_(filter)
    {
        while (!m_base_->done() && !m_filter_(*m_base_))
        {
            m_base_->next();
        }
    }

    virtual ~CursorFilter() = default;

    std::unique_ptr<base_type> copy() const override { return std::make_unique<this_type>(*this); }

    bool done() const override { return m_base_->done(); }

    pointer get_pointer() override { return m_base_->get_pointer(); }

    reference get_reference() override { return m_base_->get_reference(); }

    bool next() override
    {
        m_base_->next();

        while (!m_base_->done() && !m_filter_(*m_base_))
        {
            m_base_->next();
        }

        return !m_base_->done();
    }

protected:
    std::unique_ptr<CursorProxy<U>> m_base_;
    filter_type m_filter_;
};

// mapper

template <typename U, typename V, typename Mapper>
class CursorMapper : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;

    typedef CursorMapper<U, V, Mapper> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    typedef Mapper mapper_type;

    CursorMapper(CursorProxy<V>* base, const mapper_type& mapper) : m_base_(base), m_mapper_(mapper) {}

    CursorMapper(const this_type& other) : m_base_(other.m_base_->copy()), m_mapper_(other.m_mapper_) {}

    CursorMapper(this_type&& other) : m_base_(other.m_base_.release()), m_mapper_(other.m_mapper_) {}

    virtual ~CursorMapper() = default;

    std::unique_ptr<base_type> copy() const override { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const override { return m_base_->done(); }

    pointer get_pointer() override { return nullptr; /* m_mapper_(m_base_->get_pointer());*/ }

    reference get_reference() override { return m_mapper_(m_base_->get_reference()); }

    bool next() override { return m_base_->next(); }

protected:
    std::unique_ptr<CursorProxy<V>> m_base_;
    mapper_type m_mapper_;
};

template <typename U, typename V, typename Mapper>
CursorProxy<U>* make_mapper(CursorProxy<V>* base, Mapper const& mapper)
{
    return new CursorMapper<U, V, Mapper>(base, mapper);
}

// template <typename U>
// class CursorFilter : public CursorProxy<U>
// {
// public:
//     typedef CursorProxy<U> base_type;
//     typedef CursorFilter<U> this_type;

//     // using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;

//     CursorFilter(CursorProxy<V>* it) : m_base_(it), m_pointer_(nullptr) { update(); }

//     CursorFilter(const this_type& other) : m_base_(other.m_base_->copy().release()), m_pointer_(nullptr) { update(); }

//     CursorFilter(this_type&& other) : m_base_(other.m_base_.release()), m_pointer_(nullptr) { update(); }

//     virtual ~CursorFilter(){};

//     std::unique_ptr<base_type> copy() const override { return std::unique_ptr<base_type>(new this_type(*this)); }

//     bool done() const override { return m_base_->done(); }

//     pointer get_pointer() override { return pointer(m_pointer_.get()); }

//     reference get_reference() override { return *m_pointer_; }

//     bool next() override
//     {
//         auto res = m_base_->next();
//         update();
//         return res;
//     }

// protected:
//     std::unique_ptr<CursorProxy<V>> m_base_;
//     std::unique_ptr<value_type> m_pointer_;

//     void update()
//     {
//         if (m_base_ == nullptr || m_base_->done())
//         {
//             m_pointer_ = (nullptr);
//         }
//         else
//         {
//             m_pointer_.reset(new value_type(m_base_->get_reference()));
//         }
//     }
// };

template <typename U, typename Filter>
CursorProxy<U>* make_filter(CursorProxy<U>* base, Filter const& filter)
{
    return new CursorFilter<U, Filter>(base, filter);
}

template <typename U>
CursorProxy<U>* make_cursor_proxy() { return new CursorProxy<U, std::nullptr_t>(); }

template <typename U, typename IT, typename... Args>
auto make_cursor_proxy(const IT& ib, Args&&... args) -> std::enable_if_t<traits::is_complete<CursorProxy<U, IT>>::value, CursorProxy<U>*>
{
    return new CursorProxy<U, IT>(ib, std::forward<Args>(args)...);
}

template <typename U, typename V, typename... Args>
auto make_cursor_proxy(CursorProxy<V>* base, Args&&... args) -> std::enable_if_t<traits::is_complete<CursorProxy<U, CursorProxy<V>>>::value, CursorProxy<U>*>
{
    return new CursorProxy<U, CursorProxy<V>>(base, std::forward<Args>(args)...);
}

template <typename IT>
auto iterator_to_cursor(const IT& ib, const IT& ie) -> CursorProxy<typename std::remove_pointer_t<typename std::iterator_traits<IT>::pointer>>*
{
    typedef std::remove_pointer_t<typename std::iterator_traits<IT>::pointer> value_type;
    return new CursorProxy<value_type, IT>(ib, ie);
}

template <typename U, typename IT, typename... Args>
auto make_cursor_proxy(const IT& ib, const IT& ie, Args&&... args) -> std::enable_if_t<!traits::is_complete<CursorProxy<U, IT>>::value, CursorProxy<U>*>
{
    return make_mapper<U>(iterator_to_cursor(ib, ie), std::forward<Args>(args)...);
}
} // namespace _detail

//----------------------------------------------------------------------------------------------------------------------------------------
template <typename T>
class Cursor
{
public:
    typedef T value_type;
    typedef Cursor<value_type> cursor;
    typedef Cursor<value_type> this_type;

    typedef typename CursorProxy<value_type>::reference reference;
    typedef typename CursorProxy<value_type>::pointer pointer;
    typedef typename CursorProxy<value_type>::difference_type difference_type;


    template <typename... Args>
    Cursor(Args&&... args) : m_proxy_(_detail::make_cursor_proxy<value_type>(std::forward<Args>(args)...)) {}

    Cursor(const Cursor& other) : m_proxy_(other.m_proxy_->copy().release()) {}

    Cursor(Cursor&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Cursor() = default;

    void swap(this_type& other) { std::swap(m_proxy_, other.m_proxy_); }

    this_type operator=(const this_type& other)
    {
        Cursor(other).swap(*this);
        return *this;
    }

    bool operator==(const this_type& other) const { return m_proxy_->equal(m_proxy_.get()); }

    bool operator!=(const this_type& other) const { return m_proxy_->not_equal(m_proxy_.get()); }

    // difference_type operator-(const cursor& other) const { return m_proxy_->distance(m_proxy_.get()); }

    reference operator*() const { return m_proxy_->get_reference(); }

    pointer operator->() const { return m_proxy_->get_pointer(); }

    bool done() const { return m_proxy_->done(); }

    bool next() { return m_proxy_->next(); }

    template <typename U, typename... Args>
    Cursor<U> map(Args&&... args) const { return Cursor<U>(_detail::make_mapper<U>(m_proxy_->copy(), std::forward<Args>(args)...)); }

    template <typename... Args>
    Cursor<value_type> filter(Args&&... args) const { return Cursor<value_type>(_detail::make_filter(m_proxy_->copy(), std::forward<Args>(args)...)); }

    // private:
    std::unique_ptr<CursorProxy<value_type>> m_proxy_;
};

} // namespace db
} // namespace sp
#endif // SPDB_Cursor_h_