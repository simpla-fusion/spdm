#ifndef SPDB_Cursor_h_
#define SPDB_Cursor_h_
#include <functional>
#include <iterator>
#include <memory>

namespace sp::db
{
template <typename U, typename Enable = void>
struct cursor_traits
{
    typedef U value_type;
    typedef U& reference;
    typedef U* pointer;
    typedef ptrdiff_t difference_type;
};

template <typename U, typename V = U, typename Enable = void>
struct CursorProxy;

template <typename U>
struct CursorProxy<U>
{
    typedef CursorProxy<U, V, Enable> this_type;

    typedef U value_type;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    CursorProxy() = default;

    virtual ~CursorProxy() = default;

    virtual std::unique_ptr<CursorProxy<U>> copy() const = 0;

    virtual bool done() const { return get_pointer() == nullptr; }

    virtual bool equal(const this_type* other) const { return get_pointer() == other->get_pointer(); }

    virtual bool not_equal(const this_type* other) const { return !equal(other); }

    virtual reference get_reference() const { return *get_pointer(); }

    virtual pointer get_pointer() const = 0;

    virtual void next() = 0;

    // virtual difference_type distance(const this_type* other) const = 0;
};

template <typename U, typename V>
class CursorProxy<U, V, std::enable_if_t<!std::is_same_v<U, V>>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, V> this_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(CursorProxy<V>* it,
                const std::function<reference(typename cursor_traits<V>::reference)>& r_map,
                const std::function<pointer(typename cursor_traits<V>::pointer)>& p_map)
        : m_it_(it), m_r_map_(r_map), m_p_map_(p_map) {}

    CursorProxy(const this_type& other) : m_it_(other.m_it_->copy()), m_r_map_(other.m_r_map_), m_p_map_(other.m_p_map_) {}

    CursorProxy(this_type&& other) : m_it_(other.m_it_->release()), m_r_map_(other.m_r_map_), m_p_map_(other.m_p_map_) {}

    virtual ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() const override { return m_p_map_(m_it_->get_pointer()); }

    reference get_reference() const override { return m_r_map_(m_it_->get_reference()); }

    void next() override { m_it_->next(); }

protected:
    std::unique_ptr<CursorProxy<V>> m_it_;
    std::function<reference(typename cursor_traits<V>::reference)> m_r_map_;
    std::function<pointer(typename cursor_traits<V>::pointer)> m_p_map_;
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
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    CursorProxy(const base_iterator& ib, difference_type pos = 0) : m_base_(ib), m_end_(pos), m_pos_(0) {}

    CursorProxy(const base_iterator& ib, const base_iterator& ie) : m_base_(ib), m_end_(std::distance(ib, ie)), m_pos_(0) {}

    virtual ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() const override { return m_pos_ >= m_end_ ? nullptr : pointer(&(*m_base_) + m_pos_); }

    void next() override
    {
        if (m_pos_ < m_end_)
        {
            ++m_pos_;
        }
    }

protected:
    base_iterator m_base_;
    difference_type m_end_, m_pos_;
};

template <typename U, typename V>
class CursorProxy<U, V, std::enable_if_t<std::is_same_v<U, typename std::iterator_traits<V>::value_type>>>
    : public CursorProxy<U>
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

    CursorProxy(const iterator& ib) : CursorProxy(ib, ++iterator(ib)) {}

    virtual ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() const override { return (m_it_ == m_ie_) ? nullptr : &get_reference(); }

    reference get_reference() const override { return (*m_it_); }

    void next() override { ++m_it_; }

protected:
    iterator m_it_, m_ie_;
};

template <typename U, typename V>
class CursorProxy<U, V,
                  std::enable_if_t<
                      std::is_same_v<std::pair<const std::string, U>, typename std::iterator_traits<V>::value_type> ||
                      std::is_same_v<std::pair<const std::string, std::remove_const_t<U>>, typename std::iterator_traits<V>::value_type>>>
    : public CursorProxy<U>
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

    CursorProxy(const iterator& ib) : CursorProxy(ib, ++iterator(ib)) {}

    virtual ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    pointer get_pointer() const override { return (m_it_ == m_ie_) ? nullptr : &get_reference(); }

    reference get_reference() const override { return m_it_->second; }

    void next() override { ++m_it_; }

protected:
    iterator m_it_, m_ie_;
};

// filter
template <typename U>
class CursorProxy<U, std::function<bool(const U&)>> : public CursorProxy<U>
{
public:
    typedef U value_type;
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, std::function<bool(const U&)>> this_type;
    typedef std::function<bool(const U&)> filter_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(CursorProxy<U>* it, const filter_type filter) : m_base_(it), m_filter_(filter)
    {
        while (!m_base_->done() && !m_filter_(*m_base_))
        {
            m_base_->next();
        }
    }

    CursorProxy(std::unique_ptr<CursorProxy<U>>&& it, const filter_type filter) : CursorProxy(it.release(), filter) {}

    CursorProxy(const std::unique_ptr<CursorProxy<U>>& it, const filter_type filter) : CursorProxy(it->copy().reslease(), filter) {}

    virtual ~CursorProxy() = default;

    bool done() const { return m_base_->done(); }

    pointer get_pointer() const override { return m_base_->get_pointer(); }

    reference get_reference() const override { return m_base_->get_reference(); }

    void next() override
    {
        m_base_->next();

        if (!m_base_->done() && !m_filter_(*m_base_))
        {
            m_base_->next();
        }
    }

protected:
    std::unique_ptr<CursorProxy<U>> m_base_;
    filter_type m_filter_;
};

// mapper
template <typename U, typename V>
class CursorProxy<const U, std::function<U(const V&)>> : public CursorProxy<const U>
{
public:
    typedef U value_type;
    typedef CursorProxy<U> base_type;
    typedef std::function < U(const V&) mapper_type;

    typedef CursorProxy<const U, mapper_type> this_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(CursorProxy<const V>* it, const mapper_type filter)
        : m_base_(it), m_mapper_(filter), m_value_(m_base_ == nullptr || m_base_->done() ? nullptr : new value_type(m_mapper_(*m_base_)))
    {
    }

    CursorProxy(std::unique_ptr<CursorProxy<const U>>&& it, const filter_type filter) : CursorProxy(it.release(), filter) {}

    CursorProxy(const std::unique_ptr<CursorProxy<const U>>& it, const filter_type filter) : CursorProxy(it->copy().reslease(), filter) {}

    virtual ~CursorProxy() = default;

    bool done() const { return m_base_->done(); }

    pointer get_pointer() const override { return m_value_.get(); }

    reference get_reference() const override { return *m_value_; }

    void next() override
    {
        if (!m_base_->done())
        {
            m_base_->next();
            m_value_.reset(new value_type(m_mapper_(*m_base_)));
        }
    }

protected:
    std::unique_ptr<CursorProxy<const V>> m_base_;

    mapper_type m_mapper_;
    std::unique_ptr<const U> m_value_;
};

template <typename U, typename V>
std::unique_ptr<CursorProxy<U>> make_proxy(const V& ib, const V& ie)
{
    return std::make_unique<CursorProxy<U, V>>(ib, ie);
}
template <typename U, typename V>
std::unique_ptr<CursorProxy<U>> make_proxy(V&& ib)
{
    return std::make_unique<CursorProxy<U, std::remove_reference_t<V>>>(std::forward<V>(ib));
}

template <typename U>
std::unique_ptr<CursorProxy<U>> make_proxy(Cursor<U>&& it) { return std::move(it.m_proxy_); }

template <typename U>
std::unique_ptr<CursorProxy<U>> make_proxy(const Cursor<U>& it) { return it.m_proxy_->copy(); }

template <typename U>
std::unique_ptr<CursorProxy<U>> make_proxy(std::unique_ptr<CursorProxy<U>>&& it, Args&&... args) { return std::move(it); }

template <typename U, typename V, typename... Args>
std::unique_ptr<CursorProxy<U>> make_proxy(std::unique_ptr<CursorProxy<V>>&& it, Args&&... args) { return std::make_unique<CursorProxy<U, V>>(it.release(), std::forward<Args>(args)...); }

template <typename T>
class Cursor
{
public:
    typedef T value_type;
    typedef Cursor<value_type> cursor;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    template <typename... Args>
    Cursor(Args&&... args) : m_proxy_(make_proxy<value_type>(std::forward<Args>(args)...).release()) {}

    Cursor(const Cursor& other) : m_proxy_(other.m_proxy_->copy()) {}

    Cursor(Cursor&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Cursor() = default;

    operator bool() const { return !m_proxy_->done(); }

    bool operator==(const cursor& other) const { return m_proxy_->equal(m_proxy_.get()); }

    bool operator!=(const cursor& other) const { return m_proxy_->not_equal(m_proxy_.get()); }

    difference_type operator-(const cursor& other) const { return m_proxy_->distance(m_proxy_.get()); }

    reference operator*() const { return m_proxy_->get_reference(); }

    pointer operator->() const { return m_proxy_->get_pointer(); }

    void next() { m_proxy_->next(); }

    std::unique_ptr<CursorProxy<value_type>> m_proxy_;
};

} // namespace sp::db

#endif // SPDB_Cursor_h_