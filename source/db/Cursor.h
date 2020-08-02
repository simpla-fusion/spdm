#ifndef SPDB_Cursor_h_
#define SPDB_Cursor_h_
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

template <typename V>
struct is_cursor
{
    static const bool value = false;
};

template <typename V>
struct is_cursor<Cursor<V>>
{
    static const bool value = true;
};
template <typename V>
static const bool is_cursor_v = is_cursor<V>::value;

template <typename U, typename Enable = void>
struct cursor_traits
{
    typedef U value_type;
    typedef U& reference;
    typedef U* pointer;
    typedef ptrdiff_t difference_type;
};

namespace _detail
{
template <typename U, typename V = U, typename Enable = void>
struct CursorProxy;

template <typename U>
struct CursorProxy<U>
{
    typedef CursorProxy<U> this_type;

    typedef U value_type;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    CursorProxy() = default;

    virtual ~CursorProxy() = default;

    virtual std::unique_ptr<CursorProxy<U>> copy() const = 0;

    virtual reference get_reference() const = 0;

    virtual pointer get_pointer() const = 0;

    virtual bool next() = 0;

    virtual bool done() const { return get_pointer() == nullptr; }

    //  virtual bool equal(const this_type* other) const { return get_pointer() == other->get_pointer(); }

    //     virtual bool not_equal(const this_type* other) const { return !equal(other); }

    // virtual difference_type distance(const this_type* other) const = 0;
};

template <typename U, typename V>
class CursorProxy<U, V,
                  std::enable_if_t<                                              //
                      std::is_same_v<V, U*> ||                                   //
                      std::is_same_v<V, const U*> ||                             //
                      std::is_same_v<V, std::shared_ptr<U>> ||                   //
                      std::is_same_v<V, std::shared_ptr<std::remove_const_t<U>>> //
                      >> : public CursorProxy<U>
{

public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, V> this_type;
    typedef V base_iterator;

    typedef U value_type;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    // typedef typename cursor_traits<value_type>::difference_type difference_type;
    typedef ptrdiff_t difference_type;

    CursorProxy(const base_iterator& ib, difference_type pos = 0) : m_base_(ib), m_end_(pos), m_pos_(0) {}

    CursorProxy(const base_iterator& ib, const base_iterator& ie) : m_base_(ib), m_end_(std::distance(ib, ie)), m_pos_(0) {}

    ~CursorProxy() = default;

    bool done() const { return m_pos_ >= m_end_; }

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() const override { return m_pos_ >= m_end_ ? nullptr : pointer(&(*m_base_) + m_pos_); }

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
    difference_type m_end_, m_pos_;
};

template <typename U, typename V>
class CursorProxy<U, V,
                  std::enable_if_t<
                      std::is_same_v<U, typename std::iterator_traits<V>::value_type> || //
                      std::is_same_v<std::remove_const_t<U>, typename std::iterator_traits<V>::value_type>>> : public CursorProxy<U>
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

    ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    bool done() const override { return m_it_ == m_ie_; }

    pointer get_pointer() const override { return m_it_.operator->(); }

    reference get_reference() const override { return m_it_.operator*(); }

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

// filter
template <typename U>
class CursorProxyFilter : public CursorProxy<U>
{
public:
    typedef U value_type;
    typedef CursorProxy<U> base_type;
    typedef CursorProxyFilter<U> this_type;
    typedef std::function<bool(const U&)> filter_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;

    CursorProxyFilter(CursorProxy<U>* it, const filter_type filter) : m_base_(it), m_filter_(filter)
    {
        while (!m_base_->done() && !m_filter_(*m_base_))
        {
            m_base_->next();
        }
    }

    virtual ~CursorProxyFilter() = default;

    bool done() const { return m_base_->done(); }

    pointer get_pointer() const override { return m_base_->get_pointer(); }

    reference get_reference() const override { return m_base_->get_reference(); }

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
template <typename U, typename V, typename Enable = void>
class CursorProxyMapper;

// template <typename U, typename V>
// class CursorProxyMapper<U, V, std::enable_if_t<!std::is_convertible_v<V, U>>> : public CursorProxy<U>
// {
// public:
//     typedef CursorProxy<U> base_type;
//     typedef std::function<U(const V&)> mapper_type;

//     typedef CursorProxyMapper<U, V> this_type;

//     // using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;

//     CursorProxyMapper(CursorProxy<V>* it, const mapper_type& mapper) : m_base_(it), m_mapper_(mapper), m_pointer_(nullptr) { update(); }

//     CursorProxyMapper(CursorProxy<V>* it) : m_base_(it), m_mapper_(), m_pointer_(nullptr) { update(); }

//     CursorProxyMapper(const this_type& other) : m_base_(other.m_base_->copy().release()), m_mapper_(other.m_mapper_) { update(); }

//     CursorProxyMapper(this_type&& other) : m_base_(other.m_base_.release()), m_mapper_(other.m_mapper_) { update(); }

//     virtual ~CursorProxyMapper(){};

//     std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

//     bool done() const { return m_base_->done(); }

//     pointer get_pointer() const override { return m_pointer_; }

//     reference get_reference() const override { return *m_pointer_; }

//     bool next() override
//     {
//         auto res = m_base_->next();
//         update();
//         return res;
//     }

// protected:
//     std::unique_ptr<CursorProxy<V>> m_base_;
//     mapper_type m_mapper_;
//     pointer m_pointer_;

//     void update()
//     {
//         if (m_base_ == nullptr || m_base_->done())
//         {
//             m_pointer_ = (nullptr);
//         }
//         else
//         {
//             // m_value_=(m_mapper_(*m_base_));
//         }
//     }
// };

template <typename U, typename V>
class CursorProxyMapper<U, V, std::enable_if_t<std::is_convertible_v<V, U>>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxyMapper<U, V> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxyMapper(CursorProxy<V>* it) : m_base_(it), m_pointer_(nullptr) { update(); }

    CursorProxyMapper(const this_type& other) : m_base_(other.m_base_->copy().release()), m_pointer_(nullptr) { update(); }

    CursorProxyMapper(this_type&& other) : m_base_(other.m_base_.release()), m_pointer_(nullptr) { update(); }

    virtual ~CursorProxyMapper(){};

    std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const { return m_base_->done(); }

    pointer get_pointer() const override { return m_pointer_.get(); }

    reference get_reference() const override { return *m_pointer_; }

    bool next() override
    {
        auto res = m_base_->next();
        update();
        return res;
    }

protected:
    std::unique_ptr<CursorProxy<V>> m_base_;
    std::unique_ptr<value_type> m_pointer_;

    void update()
    {
        if (m_base_ == nullptr || m_base_->done())
        {
            m_pointer_ = (nullptr);
        }
        else
        {
            m_pointer_.reset(new value_type(*m_base_));
        }
    }
};

template <typename U, typename V>
class CursorProxyMapper<U, V,
                        std::enable_if_t<
                            std::is_same_v<std::pair<const std::string, U>, V> || //
                            std::is_same_v<std::pair<const std::string, std::remove_reference_t<U>>, V>>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;

    typedef CursorProxyMapper<U, V> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxyMapper(CursorProxy<V>* it) : m_base_(it) {}

    CursorProxyMapper(const this_type& other) : m_base_(other.m_base_->copy().release()) {}

    CursorProxyMapper(this_type&& other) : m_base_(other.m_base_.release()) {}

    virtual ~CursorProxyMapper(){};

    std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const { return m_base_->done(); }

    pointer get_pointer() const override { return &m_base_->get_pointer()->second; }

    reference get_reference() const override { return m_base_->get_pointer()->second; }

    bool next() override { return m_base_->next(); }

protected:
    std::unique_ptr<CursorProxy<V>> m_base_;
};

} // namespace _detail
//----------------------------------------------------------------------------------------------------------------------------------------
template <typename T>
class Cursor
{
public:
    typedef T value_type;
    typedef Cursor<value_type> cursor;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    // typedef typename cursor_traits<value_type>::difference_type difference_type;

    template <typename IT>
    Cursor(const IT& ib, const IT& ie)
        : m_proxy_(dynamic_cast<_detail::CursorProxy<value_type>*>(new _detail::CursorProxy<value_type, IT>(ib, ie))) {}

    template <typename V, typename... Args>
    Cursor(const Cursor<V>& other, Args&&... args)
        : m_proxy_(new _detail::CursorProxyMapper<value_type, V>(
              other.m_proxy_->copy().release(),
              std::forward<Args>(args)...)) {}

    Cursor(_detail::CursorProxy<value_type>* p) : m_proxy_(p) {}

    Cursor(const Cursor& other) : m_proxy_(other.m_proxy_->copy().release()) {}

    Cursor(Cursor&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Cursor() = default;

    // operator bool() const { return !m_proxy_->done(); }

    bool operator==(const cursor& other) const { return m_proxy_->equal(m_proxy_.get()); }

    bool operator!=(const cursor& other) const { return m_proxy_->not_equal(m_proxy_.get()); }

    // difference_type operator-(const cursor& other) const { return m_proxy_->distance(m_proxy_.get()); }

    reference operator*() const { return m_proxy_->get_reference(); }

    pointer operator->() const { return m_proxy_->get_pointer(); }

    bool next() { return m_proxy_->next(); }

    template <typename V, typename... Args>
    Cursor<V> map(const Args&&... args) const
    {
        return Cursor<V>(*this, std::forward<Args>(args)...);
    }

    template <typename Filter>
    Cursor<value_type> filter(const Filter& filter) const
    {
        return Cursor<value_type>(new _detail::CursorProxyFilter<value_type>(m_proxy_->copy().release(), filter));
    }

    std::unique_ptr<_detail::CursorProxy<value_type>> m_proxy_;
};

template <typename IT>
auto make_cursor(const IT& ib, const IT& ie) // -> Cursor<typename std::iterator_traits<IT>::value_type>
{
    return Cursor<typename std::iterator_traits<IT>::value_type>(ib, ie);
}

} // namespace db
} // namespace sp
#endif // SPDB_Cursor_h_