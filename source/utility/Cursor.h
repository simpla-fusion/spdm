#ifndef SP_Cursor_h_
#define SP_Cursor_h_
#include <functional>
#include <iterator>
#include <memory>

namespace sp
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
struct CursorProxy
{
    typedef CursorProxy<U, V, Enable> this_type;

    typedef U value_type;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    CursorProxy() = default;

    virtual ~CursorProxy() = default;

    virtual bool done() const { return get_pointer() == nullptr; }

    virtual bool equal(const this_type* other) const { return get_pointer() == other->get_pointer(); }

    virtual bool not_equal(const this_type* other) const { return !equal(other); }

    virtual reference get_reference() const { return *get_pointer(); }

    virtual pointer get_pointer() const = 0;

    virtual void next() = 0;

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
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    CursorProxy(const base_iterator& ib, difference_type pos = 0) : m_base_(ib), m_end_(pos), m_pos_(0) {}

    CursorProxy(const base_iterator& ib, const base_iterator& ie) : m_base_(ib), m_end_(std::distance(ib, ie)), m_pos_(0) {}

    virtual ~CursorProxy() = default;

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
class CursorProxy<U, V,
                  std::enable_if_t<
                      std::is_same_v<
                          U,
                          typename std::iterator_traits<V>::value_type>>>
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

    pointer get_pointer() const override { return (m_it_ == m_ie_) ? nullptr : &get_reference(); }

    reference get_reference() const override { return m_it_->second; }

    void next() override { ++m_it_; }

protected:
    iterator m_it_, m_ie_;
};

// template <typename U, typename R, typename V,
//           std::enable_if_t<std::is_same_v<typename cursor_traits<U>::refernce, R>>>
// class CursorProxy<U, std::function<R(const V&)>> : public CursorProxy<U, V>
// {
// public:
//     typedef U value_type;
//     typedef CursorProxy<U, V> base_type;

//     typedef std::function<U(const V&)> mapper_t;

//     using typename base_type::difference_type;
//     using typename base_type::pointer;
//     using typename base_type::reference;
//     using typename base_type::value_type;

//     template <typename... Args>
//     CursorProxy(const mapper_t mapper, Args&&... args) : base_type(std::forward<Args>(args)...), m_mapper_(mapper) {}

//     virtual ~CursorProxy() = default;

//     pointer get_pointer() const override = delete;

//     reference get_reference() const override { return m_mapper_(m_base_->get_reference()); }

//     void next() override { base_type::next(); }

// protected:
//     mapper_t m_mapper_;
// };

template <typename U, typename V>
CursorProxy<U>* make_proxy(const V& ib, const V& ie)
{
    return new CursorProxy<U, V>(ib, ie);
}
template <typename U, typename V>
CursorProxy<U>* make_proxy(const V& ib)
{
    return new CursorProxy<U, V>(ib);
}

template <typename TNode>
class Cursor
{
public:
    typedef TNode value_type;
    typedef Cursor<value_type> cursor;
    typedef typename cursor_traits<value_type>::reference reference;
    typedef typename cursor_traits<value_type>::pointer pointer;
    typedef typename cursor_traits<value_type>::difference_type difference_type;

    template <typename... Args>
    Cursor(Args&&... args) : m_proxy_(make_proxy<value_type>(std::forward<Args>(args)...)) {}

    ~Cursor() = default;

    operator bool() const { return !m_proxy_->done(); }

    bool operator==(const cursor& other) const { return m_proxy_->equal(m_proxy_.get()); }

    bool operator!=(const cursor& other) const { return m_proxy_->not_equal(m_proxy_.get()); }

    difference_type operator-(const cursor& other) const { return m_proxy_->distance(m_proxy_.get()); }

    reference operator*() const { return m_proxy_->get_reference(); }

    pointer operator->() const { return m_proxy_->get_pointer(); }

    void next() { m_proxy_->next(); }

private:
    std::unique_ptr<CursorProxy<value_type>> m_proxy_;
};

} // namespace sp

#endif // SP_Cursor_h_