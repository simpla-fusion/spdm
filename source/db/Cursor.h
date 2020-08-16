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

    virtual std::unique_ptr<CursorProxy<U>> copy() const { return std::make_unique<this_type>(*this); };

    virtual reference get_reference()
    {
        throw std::runtime_error("try to get reference from null object!");
        return *get_pointer();
    }

    virtual pointer get_pointer() { return nullptr; }

    virtual bool next() { return false; };

    virtual bool done() const { return true; };

    // virtual bool equal(const this_type* other) const { return get_pointer() == other->get_pointer(); }

    // virtual bool not_equal(const this_type* other) const { return !equal(other); }

    // virtual difference_type distance(const this_type* other) const = 0;
};

template <typename U, typename V>
class CursorProxy<U, V,
                  std::enable_if_t<std::is_same_v<V, std::shared_ptr<U>> ||
                                   std::is_same_v<V, std::shared_ptr<std::remove_const_t<U>>> //
                                   >>
    : public CursorProxy<U>
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

    pointer get_pointer() override { return m_pos_ >= m_end_ ? nullptr : pointer(&(*m_base_) + m_pos_); }

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
class CursorProxy<U, V, std::enable_if_t<std::is_convertible_v<typename std::iterator_traits<V>::value_type, U>>> : public CursorProxy<U>
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

    pointer get_pointer() override { return pointer(&*m_it_); }

    reference get_reference() override { return reference(*m_it_); }

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
template <typename U, typename V>
class CursorProxy<U, V, std::enable_if_t<!std::is_convertible_v<typename std::iterator_traits<V>::value_type, U>>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, V> this_type;
    typedef V iterator;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    typedef typename std::iterator_traits<V>::value_type base_value_type;
    typedef std::function<reference(const base_value_type&)> mapper_type;

    CursorProxy(const iterator& ib, const iterator& ie, const mapper_type& mapper) : m_it_(ib), m_ie_(ie), m_mapper_(mapper) {}

    ~CursorProxy() = default;

    std::unique_ptr<CursorProxy<U>> copy() const override { return std::make_unique<this_type>(*this); }

    bool done() const override { return m_it_ == m_ie_; }

    pointer get_pointer() override
    {
        NOT_IMPLEMENTED;
        return pointer(nullptr);
    }

    reference get_reference() override { return m_mapper_(*m_it_); }

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
    mapper_type m_mapper_;
};

// filter

template <typename U>
class CursorProxy<U, Cursor<U>> : public CursorProxy<U>
{
public:
    typedef U value_type;
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, Cursor<U>> this_type;
    typedef std::function<bool(const U&)> filter_type;

    using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;

    CursorProxy(const Cursor<U>& it, const filter_type filter) : m_base_(it.m_proxy_->copy()), m_filter_(filter)
    {
        while (!m_base_->done() && !m_filter_(*m_base_))
        {
            m_base_->next();
        }
    }

    virtual ~CursorProxy() = default;

    bool done() const { return m_base_->done(); }

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

template <typename U, typename V>
class CursorProxy<U, Cursor<V>,
                  std::enable_if_t<!std::is_convertible_v<V, U>>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;

    typedef CursorProxy<U, Cursor<V>> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    typedef std::function<reference(const V&)> mapper_type;

    CursorProxy(const Cursor<V>& it, const mapper_type& mapper) : m_base_(it.m_proxy_->copy()), m_mapper_(mapper) {}

    CursorProxy(const Cursor<V>& it) : m_base_(it->m_proxy_->copy()), m_mapper_() {}

    CursorProxy(const this_type& other) : m_base_(other.m_base_->copy().release()) {}

    CursorProxy(this_type&& other) : m_base_(other.m_base_.release()), m_mapper_(other.m_mapper_) {}

    virtual ~CursorProxy(){};

    std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const { return m_base_->done(); }

    pointer get_pointer() override { return nullptr; /* m_mapper_(m_base_->get_pointer());*/ }

    reference get_reference() override { return m_mapper_(m_base_->get_reference()); }

    bool next() override { return m_base_->next(); }

protected:
    std::unique_ptr<CursorProxy<V>> m_base_;
    mapper_type m_mapper_;
};

template <typename U, typename V>
class CursorProxy<U, Cursor<V>,
                  std::enable_if_t<!std::is_same_v<V, U> && std::is_convertible_v<V, U>>> : public CursorProxy<U>
{
public:
    typedef CursorProxy<U> base_type;
    typedef CursorProxy<U, V> this_type;

    // using typename base_type::difference_type;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::value_type;

    CursorProxy(CursorProxy<V>* it) : m_base_(it), m_pointer_(nullptr) { update(); }

    CursorProxy(const this_type& other) : m_base_(other.m_base_->copy().release()), m_pointer_(nullptr) { update(); }

    CursorProxy(this_type&& other) : m_base_(other.m_base_.release()), m_pointer_(nullptr) { update(); }

    virtual ~CursorProxy(){};

    std::unique_ptr<base_type> copy() const { return std::unique_ptr<base_type>(new this_type(*this)); }

    bool done() const { return m_base_->done(); }

    pointer get_pointer() override { return pointer(m_pointer_.get()); }

    reference get_reference() override { return *m_pointer_; }

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
            m_pointer_.reset(new value_type(m_base_->get_reference()));
        }
    }
};

//----------------------------------------------------------------------------------------------------------------------------------------
template <typename T>
class Cursor
{

public:
    typedef T value_type;
    typedef Cursor<value_type> cursor;
    typedef Cursor<value_type> this_type;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    // typedef typename cursor_traits<value_type>::difference_type difference_type;

    explicit Cursor(CursorProxy<value_type>* p = nullptr) : m_proxy_(p != nullptr ? p : new CursorProxy<value_type>) {}

    template <typename U, typename... Args>
    Cursor(const U& it, Args&&... args) : m_proxy_(new CursorProxy<T, U>(it, std::forward<Args>(args)...)) {}

    Cursor(const Cursor& other) : m_proxy_(other.m_proxy_->copy().release()) {}

    Cursor(Cursor&& other) : m_proxy_(other.m_proxy_.release()) {}

    ~Cursor() = default;

    // operator bool() const { return !m_proxy_->done(); }

    void swap(cursor& other)
    {
        std::swap(m_proxy_, other.m_proxy_);
    }

    cursor operator=(const cursor& other)
    {
        Cursor(other).swap(*this);
        return *this;
    }

    bool operator==(const cursor& other) const { return m_proxy_->equal(m_proxy_.get()); }

    bool operator!=(const cursor& other) const { return m_proxy_->not_equal(m_proxy_.get()); }

    // difference_type operator-(const cursor& other) const { return m_proxy_->distance(m_proxy_.get()); }

    reference operator*() const { return m_proxy_->get_reference(); }

    pointer operator->() const { return m_proxy_->get_pointer(); }

    bool done() const { return m_proxy_->done(); }

    bool next() { return m_proxy_->next(); }

    template <typename U, typename... Args>
    Cursor<U> map(Args&&... args) const { return Cursor<U>(*this, std::forward<Args>(args)...); }

    template <typename Filter>
    Cursor<value_type> filter(const Filter& filter) const { return Cursor<value_type>(*this, filter); }

    // private:
    std::unique_ptr<CursorProxy<value_type>> m_proxy_;
};

} // namespace db
} // namespace sp
#endif // SPDB_Cursor_h_