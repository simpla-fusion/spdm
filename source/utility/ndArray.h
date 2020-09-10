//
// Created by salmon on 16-12-28.
//

#ifndef SP_ARRAY_H
#define SP_ARRAY_H

#include "ExpressionTemplate.h"
#include "TypeTraits.h"
#include "nTuple.h"
#include <initializer_list>
#include <limits>
#include <memory>
#include <tuple>

namespace sp
{

template <unsigned int NDIM>
class ZSFC
{
    typedef ZSFC<NDIM> this_type;

public:
    static constexpr unsigned int ndim = NDIM;

    typedef std::ptrdiff_t difference_type;
    typedef std::size_t size_type;
    typedef nTuple<difference_type, ndim> index_tuple;
    typedef nTuple<size_type, ndim> size_tuple;

public:
    nTuple<size_type, ndim> m_count_;
    nTuple<size_type, ndim> m_strides_;
    size_type m_offset_ = 0;

    /** memory shape*/

    /** index*/
    ZSFC()
    {
        m_count_ = 0;
        m_strides_ = 1;
        m_offset_ = 0;
    }

    ~ZSFC() = default;

    template <typename I, unsigned int O>
    ZSFC(nTupleBasic<I, O, ndim> d) { reshape(d); }

    ZSFC(ZSFC const& other) : m_count_(other.m_count_), m_strides_(other.m_strides_), m_offset_(other.m_offset_) {}

    ZSFC(ZSFC&& other) : m_count_(std::move(other.m_count_)), m_strides_(std::move(other.m_strides_)), m_offset_(other.m_offset_) {}

    void swap(this_type& other)
    {
        m_count_.swap(other.m_count_);
        m_strides_.swap(other.m_strides_);
        m_offset_.swap(other.m_offset_);
    }

    template <typename _UInt>
    ZSFC(std::initializer_list<_UInt> const& d) : ZSFC() { reshape(d); }

    template <typename _UInt>
    void reshape(std::initializer_list<_UInt> const& d, bool is_slow_first = true)
    {
        m_count_ = 1;
        m_count_ = d;
        reshape(m_count_, is_slow_first);
    }

    template <typename D>
    void reshape(D d, bool is_slow_first = true)
    {
        m_count_ = d;
        if (is_slow_first)
        {
            m_strides_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i)
            {
                m_strides_[i] = m_strides_[i + 1] * m_count_[i + 1];
            }
        }
        else
        {
            m_strides_[0] = 1;
            for (int i = 1; i < ndim - 1; ++i)
            {
                m_strides_[i] = m_strides_[i - 1] * m_count_[i - 1];
            }
        }
    }

    auto slice() const { return this_type(*this); }

    template <typename I>
    size_type slice(nTuple<I, ndim> const& s) const { return inner_product(m_strides_, s) + m_offset_; }

    template <typename... Args>
    auto slice(Args&&... args) const
        -> std::enable_if_t<(ndim == sizeof...(Args) && traits::is_integral<Args...>::value), size_type>
    {
        return slice(nTuple<difference_type, ndim>{std::forward<Args>(args)...});
    }

    template <typename I>
    auto slice(I const& s) const -> std::enable_if_t<(ndim > 1 && std::is_integral<I>::value), ZSFC<ndim - 1>>
    {
        ZSFC<ndim - 1> res;
        res.m_count_ = &m_count_[1];
        res.m_strides_ = &m_strides_[1];
        res.m_offset_ += s * m_strides_[0];
        return res;
    }

    template <typename I, unsigned int N>
    auto slice(nTuple<I, N> const& s) const -> std::enable_if_t<(N < ndim), ZSFC<ndim - N>>
    {
        ZSFC<ndim - N> res;
        return res;
    }

    template <typename I>
    ZSFC<ndim> slice(nTuple<I, ndim, 3> const& s) const
    {
        ZSFC<ndim> res;
        return std::move(res);
    }

    template <typename T, typename... Args>
    auto slice(T const& first, Args&&... args) const -> std::enable_if_t<
        (ndim > 1 && traits::sum<std::size_t, (traits::is_integral<Args>::value ? 1 : 0)...>::value < ndim),
        ZSFC<ndim - traits::sum<std::size_t, (traits::is_integral<Args>::value ? 1 : 0)...>::value>>
    {
        ZSFC<ndim - traits::sum<std::size_t, (traits::is_integral<Args>::value ? 1 : 0)...>::value> res;
        return res;
    }

    template <typename I>
    auto slice(std::initializer_list<I> const& list) const
    {
        nTuple<difference_type, ndim> d;
        d = list;
        return std::move(slice(d));
    }

    template <typename I>
    auto slice(std::initializer_list<std::initializer_list<I>> const& list) const
    {
        this_type res(*this);
        int s = 0;
        for (auto const& item : list)
        {
            if (s >= ndim)
            {
                break;
            }
            nTuple<difference_type, 3> t{0, static_cast<difference_type>(m_count_[s]), 1};
            t = item;
            res.m_offset_ += t[0] * res.m_strides_[s];
            res.m_strides_[s] *= t[2];
            res.m_count_[s] = (t[1] - t[0]) / t[2];
            ++s;
        }
        return std::move(res);
    }

    auto shift() const { return this_type(*this); }

    template <typename D>
    auto shift(D const& d) const
    {
        this_type res(*this);
        index_tuple s;
        s = d;
        res.m_offset_ += inner_product(m_strides_, s);
        return std::move(res);
    }

    template <typename I>
    auto shift(std::initializer_list<I> const& list) const
    {
        nTuple<difference_type, ndim> d;
        d = list;
        return std::move(shift(d));
    }

    bool empty() const { return size() == 0; }

    size_type size() const { return static_cast<size_type>(product_t(m_count_)); }

    auto const& count() const { return m_count_; }

    auto const& stride() const { return m_strides_; }

    auto offset() const { return m_offset_; }

    struct iterator
    {
    public:
        typedef nTuple<difference_type, ndim> value_type;
        typedef std::random_access_iterator_tag iterator_category;

    private:
        ZSFC<ndim> m_sfc_;
        difference_type m_pos_ = 0;
        nTuple<size_type, ndim> m_strides_;

    public:
        iterator() = default;
        iterator(iterator const& other) : m_sfc_(other.m_sfc_), m_pos_(other.m_pos_), m_strides_(other.m_strides_){};
        iterator(iterator&& other)
            : m_sfc_(std::move(other.m_sfc_)), m_pos_(other.m_pos_), m_strides_(other.m_strides_){};
        iterator(ZSFC<ndim> const& sfc, size_type pos = 0) : m_sfc_(sfc), m_pos_(pos)
        {
            m_strides_[ndim - 1] = 1;
            for (int i = ndim - 2; i >= 0; --i)
            {
                m_strides_[i] = m_sfc_.m_count_[i + 1] * m_strides_[i + 1];
            }
        }
        ~iterator() = default;
        void swap(iterator& other)
        {
            m_sfc_.swap(other.m_sfc_);
            m_strides_.swap(other.m_strides_);
            std::swap(m_pos_, other.m_pos_);
        }
        iterator& operator++()
        {
            next();
            return *this;
        }
        iterator operator++(int)
        {
            iterator tmp(*this);
            next();
            return tmp;
        }
        iterator& operator--()
        {
            prev();
            return *this;
        }
        iterator operator--(int)
        {
            iterator tmp(*this);
            prev();
            return tmp;
        }
        iterator operator+(difference_type n)
        {
            iterator tmp(*this);
            tmp += n;
            return std::move(tmp);
        }
        iterator operator-(difference_type n)
        {
            iterator tmp(*this);
            tmp -= n;
            return std::move(tmp);
        }
        iterator& operator+=(difference_type n)
        {
            next(n);
            return *this;
        }
        iterator& operator-=(difference_type n)
        {
            prev(n);
            return *this;
        }
        bool operator==(iterator const& other) { return m_pos_ == other.m_pos_; }
        bool operator!=(iterator const& other) { return m_pos_ != other.m_pos_; }

        value_type operator*() const
        {
            value_type res;
            res = ((m_pos_ / m_strides_) % m_sfc_.m_count_);
            return std::move(res);
        }
        auto position() const { return m_sfc_.slice(operator*()); }
        auto distance() const { return m_pos_; }
        auto distance(iterator const& other) const { return distance() - other.distance(); }
        auto operator-(iterator const& other) const { return distance(other); }

    private:
        void next(difference_type n = 1) { m_pos_ += n; }
        void prev(difference_type n = 1) { m_pos_ -= n; }
    };

    typedef this_type range_type;

    range_type range() const { return range_type(*this); }

    range_type index_range() const { return range_type(*this); }

    range_type operator&&(range_type const&) const { return *this; }

    iterator begin() const { return iterator{*this, 0}; }

    iterator end() const { return iterator{*this, size()}; }
};

template <unsigned int NDIM>
constexpr unsigned int ZSFC<NDIM>::ndim;

namespace utility
{
template <typename TRange, typename TFun>
void foreach_range(TRange const& range, TFun const& fun)
{
    for (decltype(auto) v : range)
    {
        fun(v);
    }
}
template <typename TRange, typename TReduction, typename TFun>
auto reduce_range(TRange const& range, TReduction const& reduction, TFun const& fun)
{
    auto it = range.begin();
    auto ie = range.end();
    auto res = fun(*it);
    ++it;
    for (; it != ie; ++it)
    {
        res = reduction(res, fun(*it));
    }
    return res;
}
} // namespace utility

template <typename...>
class Expression;

template <typename V, typename SFC = ZSFC<3>>
class Array;

namespace traits
{

template <typename V, typename SFC>
struct number_of_dimensions<Array<V, SFC>> : public std::integral_constant<unsigned int, SFC::ndim>
{
};

template <typename V, typename SFC>
struct remove_all_dimensions<Array<V, SFC>>
{
    typedef V type;
};

template <typename V, typename SFC>
struct add_reference<Array<V, SFC>>
{
    typedef Array<V, SFC> const& type;
};
template <typename V, typename SFC>
struct remove_all_extents<Array<V, SFC>>
{
    typedef V type;
};

template <typename V>
struct is_array : public std::false_type
{
};
template <typename V>
struct is_array<const V> : public std::integral_constant<bool, is_array<V>::value>
{
};
template <typename V>
struct is_array<const V&> : public std::integral_constant<bool, is_array<V>::value>
{
};
template <typename V>
struct is_array<V&> : public std::integral_constant<bool, is_array<V>::value>
{
};

template <typename V, typename SFC>
struct is_array<Array<V, SFC>> : public std::true_type
{
};
} // namespace traits

namespace calculus
{
template <typename T0>
auto get_index_range(T0 const& first, std::enable_if_t<traits::is_array<T0>::value>* _p = nullptr)
{
    return first.index_range();
}
template <typename T0, typename T1>
auto get_index_range(T0 const& first, T1 const& second,
                     std::enable_if_t<traits::is_array<T0>::value && !traits::is_array<T1>::value>* _p = nullptr)
{
    return std::move(first.index_range());
}
template <typename T0, typename T1>
auto get_index_range(T0 const& first, T1 const& second,
                     std::enable_if_t<!traits::is_array<T0>::value && traits::is_array<T1>::value>* _p = nullptr)
{
    return std::move(first.index_range());
}
template <typename T0, typename T1>
auto get_index_range(T0 const& first, T1 const& second,
                     std::enable_if_t<traits::is_array<T0>::value && traits::is_array<T1>::value>* _p = nullptr)
{
    return first.index_range() && first.index_range();
}
template <typename T0, typename... Others>
auto get_index_range(T0 const& first, Others&&... others)
{
    return get_index_range(first) && get_index_range(std::forward<Others>(others)...);
}

template <typename... Args, std::size_t... I>
auto get_index_range_helper(std::tuple<Args...> const& expr, std::index_sequence<I...>)
{
    return get_index_range(std::get<I>(expr)...);
}
template <typename... Args, std::size_t... I>
auto get_index_range(std::tuple<Args...> const& expr)
{
    return get_index_range_helper(expr, std::index_sequence_for<Args...>());
}
template <typename T0>
auto get_index_range(T0 const& expr, std::enable_if_t<traits::is_expression<T0>::value>* _p = nullptr)
{
    return get_index_range(expr.m_args_);
}
template <typename V, typename IDX, typename SFINAE = void>
struct IndexedGetHelper;

template <typename T, typename IDX>
struct IndexedGetHelper<T, IDX, std::enable_if_t<!(traits::is_array<traits::remove_cvref_t<T>>::value || traits::is_expression<traits::remove_cvref_t<T>>::value)>>
{
    template <typename TExpr, typename I>
    static decltype(auto) eval(TExpr& v, I const& idx)
    {
        return v;
    }
};
template <typename T, typename IDX>
struct IndexedGetHelper<T, IDX, std::enable_if_t<traits::is_array<traits::remove_cvref_t<T>>::value>>
{
    template <typename TExpr, typename I>
    static decltype(auto) eval(TExpr& v, I const& idx)
    {
        return v[idx];
    }
};

template <typename TExpr, typename IDX>
decltype(auto) get_idx(TExpr& expr, IDX const& idx) { return IndexedGetHelper<traits::remove_cvref_t<TExpr>, IDX>::eval(expr, idx); }

template <typename T, typename IDX>
struct IndexedGetHelper<T, IDX, std::enable_if_t<traits::is_expression<traits::remove_cvref_t<T>>::value>>
{
    template <typename TExpr, typename I, std::size_t... N>
    static decltype(auto) helper(TExpr const& expr, I const& idx, std::index_sequence<N...>)
    {
        return expr.m_op_(get_idx(std::get<N>(expr.m_args_), idx)...);
    }
    template <typename TOP, typename... Args, typename I>
    static decltype(auto) eval(Expression<TOP, Args...> const& expr, I const& idx)
    {
        return helper(expr, idx, std::index_sequence_for<Args...>());
    }
};

template <typename TReduction, typename TExpr, typename SFINAE>
struct ReduceExpressionHelper;

template <typename TReduction, typename TExpr>
struct ReduceExpressionHelper<TReduction, TExpr, std::enable_if_t<(traits::rank<TExpr>::value == 0 && traits::number_of_dimensions<TExpr>::value > 0)>>
{
    template <typename E>
    static auto eval(E const& expr)
    {
        return utility::reduce_range(get_index_range(expr), TReduction(),
                                     [&](auto const& idx) { return get_idx(expr, idx); });
    }
};

template <typename TL, typename TR, typename SFINAE>
struct EvaluateExpressionHelper;

template <typename TLHS, typename TRHS>
struct EvaluateExpressionHelper<TLHS, TRHS, std::enable_if_t<traits::is_array<TLHS>::value>>
{
    template <typename R, typename E>
    static int eval(R& lhs, E const& rhs)
    {
        auto r = get_index_range(lhs, rhs);
        for (decltype(auto) idx : r)
        {
            get_idx(lhs, idx) = get_idx(rhs, idx);
        }
        return 1;
    }
};

} // namespace calculus

template <typename V, typename SFC>
auto make_array(std::shared_ptr<V> d, SFC sfc)
{
    return Array<V, traits::remove_cvref_t<SFC>>(std::move(d), std::move(sfc));
};

template <typename V, typename SFC>
class Array
{

public:
    typedef V value_type;

    typedef SFC sfc_type;

    typedef Array<V, SFC> this_type;

    static constexpr value_type s_nan = std::numeric_limits<value_type>::signaling_NaN();

    static value_type m_null_;

    typedef Array<value_type, sfc_type> array_type;

    typedef Array<std::add_const_t<value_type>, sfc_type> const_array_type;

    static constexpr unsigned int ndim = sfc_type::ndim;

private:
    std::shared_ptr<value_type> m_data_ = nullptr;
    sfc_type m_sfc_;

public:
    Array() = default;

    ~Array() = default;

    Array(this_type const& other) : m_sfc_(other.m_sfc_), m_data_(other.m_data_) {}

    Array(this_type&& other) noexcept : m_sfc_(std::move(other.m_sfc_)), m_data_(std::move(other.m_data_)) {}

    explicit Array(sfc_type sfc) : m_sfc_(std::move(sfc)), m_data_(nullptr) {}

    explicit Array(sfc_type&& sfc) : m_sfc_(std::move(sfc)), m_data_(nullptr) {}

    explicit Array(std::shared_ptr<value_type> d, sfc_type sfc) : m_sfc_(std::move(sfc)), m_data_(std::move(d)) {}

    template <typename... Args>
    explicit Array(Args&&... args) : m_sfc_(std::forward<Args>(args)...) {}

    template <typename... Args>
    explicit Array(std::shared_ptr<value_type> d, Args&&... args) : m_sfc_(std::forward<Args>(args)...), m_data_(std::move(d)) {}

    template <typename _UInt>
    Array(std::initializer_list<_UInt> const& d) : m_sfc_(d) {}

    Array(this_type& other, tags::split) : m_sfc_(other.m_sfc_, tags::split()), m_data_(other.m_data_) {}

    void swap(this_type& other)
    {
        std::swap(m_data_, other.m_data_);
        m_sfc_.swap(other.m_sfc_);
    }

    auto split(unsigned int left = 1, unsigned int right = 1) { return this_type(*this, tags::split(left, right)); }

    auto duplicate() const { return new this_type(dynamic_cast<sfc_type const&>(*this)); }

    auto copy() const { return new this_type(*this); }

    auto index_range() const { return m_sfc_.range(); }

    auto size() const { return m_sfc_.size(); }

    auto size_in_byte() const { return m_sfc_.size() * sizeof(value_type); }

    auto value_size_in_byte() const { return sizeof(value_type); }

    auto empty() const { return m_sfc_.empty(); }

    auto is_allocated() const { return m_data_ != nullptr; }

    void reset() { reset(nullptr); }

    template <typename... Args>
    void reset(std::shared_ptr<value_type> d, Args&&... args)
    {
        m_data_ = std::move(d);
        m_sfc_.reshape(std::forward<Args>(args)...);
    }

    template <typename _UInt>
    void reshape(std::initializer_list<_UInt> const& d, bool is_slow_first = true)
    {
        m_sfc_.reshape(d, is_slow_first);
    }

    template <typename D>
    void reshape(D d, bool is_slow_first = true)
    {
        m_sfc_.reshape(std::move(d), is_slow_first);
    }

    auto data() { return m_data_; }

    auto data() const { return m_data_; }

    auto get() { return m_data_.get() + m_sfc_.offset(); }

    auto get() const { return m_data_.get() + m_sfc_.offset(); }

    auto const& sfc() const { return m_sfc_; }

    auto count() const { return m_sfc_.m_count_; }

    void clear() { fill(0); }

    this_type& operator=(this_type const& other)
    {
        alloc();
        calculus::evaluate_expression(*this, other);
        return *this;
    }

    this_type& operator=(this_type&& other)
    {
        this_type(std::move(other)).swap(*this);
        return *this;
    }

    template <typename Other>
    this_type& operator=(Other const& other)
    {
        alloc();
        calculus::evaluate_expression(*this, other);
        return (*this);
    }

    this_type& operator=(traits::make_nested_initializer_list<value_type, sfc_type::ndim> const& list)
    {
        alloc();
        calculus::evaluate_expression(*this, list);
        return *this;
    }

    template <typename S>
    decltype(auto) get_value(S s, std::enable_if_t<std::is_integral<S>::value>* _p = nullptr)
    {
        return m_data_.get()[s];
    }

    template <typename S>
    decltype(auto) get_value(S s, std::enable_if_t<std::is_integral<S>::value>* _p = nullptr) const
    {
        return m_data_.get()[s];
    }

    template <typename S>
    decltype(auto) get_value(S&& s, std::enable_if_t<!std::is_integral<traits::remove_cvref_t<S>>::value>* _p = nullptr)
    {
        return make_array(m_data_, std::forward<S>(s));
    }

    template <typename S>
    decltype(auto) get_value(S&& s, std::enable_if_t<!std::is_integral<traits::remove_cvref_t<S>>::value>* _p = nullptr) const
    {
        return make_array(m_data_, std::forward<S>(s));
    }

    template <typename... Args>
    decltype(auto) get_value(Args&&... args) { return get_value(m_sfc_.slice(std::forward<Args>(args)...)); }

    template <typename... Args>
    decltype(auto) get_value(Args&&... args) const { return get_value(m_sfc_.slice(std::forward<Args>(args)...)); }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) { return get_value(m_sfc_.slice(std::forward<Args>(args)...)); }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const { return get_value(m_sfc_.slice(std::forward<Args>(args)...)); }

    template <typename I>
    decltype(auto) operator[](std::initializer_list<I> const& s) { return get_value(m_sfc_.slice(s)); }

    template <typename I>
    decltype(auto) operator[](std::initializer_list<I> const& s) const { return get_value(m_sfc_.slice(s)); }

    template <typename I>
    decltype(auto) operator[](std::initializer_list<std::initializer_list<I>> const& s) { return get_value(m_sfc_.slice(s)); }

    template <typename I>
    decltype(auto) operator[](std::initializer_list<std::initializer_list<I>> const& s) const { return get_value(m_sfc_.slice(s)); }

    template <typename IDX>
    decltype(auto) operator[](IDX const& pos) { return get_value(m_sfc_.slice(pos)); }

    template <typename IDX>
    decltype(auto) operator[](IDX const& pos) const { return get_value(m_sfc_.slice(pos)); }

    template <typename... Args>
    decltype(auto) at(Args&&... args) { return get_value(m_sfc_.slice_with_boundcheck(std::forward<Args>(args)...)); }

    template <typename... Args>
    decltype(auto) at(Args&&... args) const { return get_value(m_sfc_.slice_with_boundcheck(std::forward<Args>(args)...)); }

    auto slice() { return this_type(m_data_, m_sfc_); }

    auto slice() const { return this_type(m_data_, m_sfc_); }

    template <typename I>
    decltype(auto) slice(std::initializer_list<I> const& s) { return get_value(m_sfc_.slice(s)); }

    template <typename I>
    decltype(auto) slice(std::initializer_list<I> const& s) const { return get_value(m_sfc_.slice(s)); }

    template <typename I>
    decltype(auto) slice(std::initializer_list<std::initializer_list<I>> const& s) { return get_value(m_sfc_.slice(s)); }

    template <typename I>
    decltype(auto) slice(std::initializer_list<std::initializer_list<I>> const& s) const { return get_value(m_sfc_.slice(s)); }

    template <typename... Args>
    decltype(auto) slice(Args&&... args) { return get_value(m_sfc_.slice(std::forward<Args>(args)...)); }

    template <typename... Args>
    decltype(auto) slice(Args&&... args) const { return get_value(m_sfc_.slice(std::forward<Args>(args)...)); }

    auto shift() { return make_array(m_data_, m_sfc_); }

    auto shift() const { return make_array(m_data_, m_sfc_); }

    template <typename I>
    decltype(auto) shift(std::initializer_list<I> const& d) const { return make_array(m_data_, m_sfc_.shift(d)); }

    template <typename... Args>
    decltype(auto) shift(Args&&... args) const { return make_array(m_data_, m_sfc_.shift(std::forward<Args>(args)...)); }

    void initialize()
    {
        alloc();
#ifndef SPDB_ARRAY_INITIALIZE_VALUE
#elif SP_ARRAY_INITIALIZE_VALUE == SP_SNaN
        fill(std::numeric_limits<V>::signaling_NaN());
#elif SP_ARRAY_INITIALIZE_VALUE == SP_QNaN
        fill(std::numeric_limits<V>::quiet_NaN());
#elif SP_ARRAY_INITIALIZE_VALUE == SP_DENORM_MIN
        fill(std::numeric_limits<V>::denorm_min());
#else
        fill(0);
#endif
    }

    void alloc()
    {
        if (m_data_ == nullptr)
        {
            assert(m_sfc_.size() > 0);
            m_data_ = std::shared_ptr<value_type>(new value_type[m_sfc_.size()]);
        }
    }

    void free() { m_data_.reset(); }

    static constexpr std::type_info const& value_type_info() { return typeid(value_type); };

    void fill_nan() { fill(std::numeric_limits<value_type>::signaling_NaN()); }

    void fill_zero() { fill(0); }

    void fill(value_type v)
    {
        alloc();
        for (auto& item : *this)
        {
            item = v;
        }
    }

    void fill(value_type v, value_type inc)
    {
        alloc();
        for (auto& item : *this)
        {
            item = v;
            v += inc;
        }
    }

    template <typename U>
    struct IteratorBasic : public sfc_type::iterator
    {
        typedef typename sfc_type::iterator base_type;
        typedef IteratorBasic<U> this_type;
        typedef V value_type;
        typedef V* pointer;
        typedef V& reference;

        value_type* m_data_;

        IteratorBasic(value_type* d, base_type const& idx_it) : base_type(idx_it), m_data_(d) {}
        IteratorBasic(IteratorBasic const& other) : base_type(other), m_data_(other.m_data_) {}
        IteratorBasic(IteratorBasic&& other) : base_type(std::move(other)), m_data_(other.m_data_) {}
        ~IteratorBasic() = default;
        IteratorBasic& operator=(IteratorBasic const& other)
        {
            IteratorBasic(other).swap(*this);
            return *this;
        }
        void swap(IteratorBasic& other)
        {
            std::swap(m_data_, other.m_data_);
            sfc_type::iterator::swap(other);
        }
        bool operator==(this_type const& other) { return m_data_ == other.m_data_ && base_type::operator==(other); }
        bool operator!=(this_type const& other) { return m_data_ != other.m_data_ || base_type::operator!=(other); }

        reference operator*() { return m_data_[base_type::position()]; }
        pointer operator->() { return &m_data_[base_type::position()]; }
    };

    typedef IteratorBasic<value_type> iterator;
    
    typedef IteratorBasic<const value_type> const_iterator;

    iterator begin()
    {
        alloc();
        return iterator(m_data_.get(), m_sfc_.begin());
    }

    iterator end()
    {
        alloc();
        return iterator(m_data_.get(), m_sfc_.end());
    }

    const_iterator begin() const
    {
        assert(is_allocated());
        return const_iterator(m_data_.get(), m_sfc_.begin());
    }

    const_iterator end() const
    {
        assert(is_allocated());
        return const_iterator(m_data_.get(), m_sfc_.end());
    }
};

template <typename V, unsigned int NDIM = 1>
using ndArray = Array<V, ZSFC<NDIM>>;

template <typename V, unsigned int NDIM>
using VecNdArray = nTuple<Array<V, ZSFC<NDIM>>, 3>;

#define DEFINE_NTUPLE_FOREACH_MEMBER_METHOD(_GLOBAL_FUN_, _MEMBER_FUN_NAME_)                                    \
    namespace detail                                                                                            \
    {                                                                                                           \
    HAS_MEMBER_FUNCTION(_MEMBER_FUN_NAME_)                                                                      \
    }                                                                                                           \
    template <typename V, typename... Args>                                                                     \
    auto _GLOBAL_FUN_(V& v, Args&&... args)                                                                     \
        ->std::enable_if_t<traits::is_array<V>::value &&                                                        \
                           detail::has_member_function_##_MEMBER_FUN_NAME_<V, Args...>::value>                  \
    {                                                                                                           \
        v._MEMBER_FUN_NAME_(std::forward<Args>(args)...);                                                       \
    };                                                                                                          \
    template <typename V, typename I>                                                                           \
    auto _GLOBAL_FUN_(V& v, std::initializer_list<I> const& list)                                               \
        ->std::enable_if_t<traits::is_array<V>::value &&                                                        \
                           detail::has_member_function_##_MEMBER_FUN_NAME_<V, std::initializer_list<I>>::value> \
    {                                                                                                           \
        v._MEMBER_FUN_NAME_(list);                                                                              \
    };                                                                                                          \
    template <typename V, typename I>                                                                           \
    auto _GLOBAL_FUN_(V& v, std::initializer_list<std::initializer_list<I>> const& list)                        \
        ->std::enable_if_t<traits::is_array<V>::value &&                                                        \
                           detail::has_member_function_##_MEMBER_FUN_NAME_<                                     \
                               V, std::initializer_list<std::initializer_list<I>>>::value>                      \
    {                                                                                                           \
        v._MEMBER_FUN_NAME_(list);                                                                              \
    };                                                                                                          \
    template <typename V, unsigned int... N, typename... Args>                                                  \
    auto _GLOBAL_FUN_(nTuple<V, N...>& v, Args&&... args)                                                       \
        ->std::enable_if_t<traits::is_array<V>::value &&                                                        \
                           detail::has_member_function_##_MEMBER_FUN_NAME_<V, Args...>::value>                  \
    {                                                                                                           \
        for (auto& item : v)                                                                                    \
        {                                                                                                       \
            _GLOBAL_FUN_(item, std::forward<Args>(args)...);                                                    \
        }                                                                                                       \
    };                                                                                                          \
    template <typename V, unsigned int... N, typename I>                                                        \
    auto _GLOBAL_FUN_(nTuple<V, N...>& v, std::initializer_list<I> const& list)                                 \
        ->std::enable_if_t<traits::is_array<V>::value &&                                                        \
                           detail::has_member_function_##_MEMBER_FUN_NAME_<V, std::initializer_list<I>>::value> \
    {                                                                                                           \
        for (auto& item : v)                                                                                    \
        {                                                                                                       \
            _GLOBAL_FUN_(item, list);                                                                           \
        }                                                                                                       \
    };                                                                                                          \
    template <typename V, unsigned int... N, typename I>                                                        \
    auto _GLOBAL_FUN_(nTuple<V, N...>& v, std::initializer_list<std::initializer_list<I>> const& list)          \
        ->std::enable_if_t<traits::is_array<V>::value &&                                                        \
                           detail::has_member_function_##_MEMBER_FUN_NAME_<                                     \
                               V, std::initializer_list<std::initializer_list<I>>>::value>                      \
    {                                                                                                           \
        for (auto& item : v)                                                                                    \
        {                                                                                                       \
            _GLOBAL_FUN_(item, list);                                                                           \
        }                                                                                                       \
    };

#define DEFINE_NTUPLE_FOREACH_MEMBER_FUNCTION(_GLOBAL_FUN_, _MEMBER_FUN_NAME_)                                        \
    namespace detail                                                                                                  \
    {                                                                                                                 \
    HAS_MEMBER_FUNCTION(_MEMBER_FUN_NAME_)                                                                            \
    }                                                                                                                 \
    template <typename V, typename... Args>                                                                           \
    auto _GLOBAL_FUN_(V&& v, Args&&... args)                                                                          \
        ->std::enable_if_t<(traits::is_array<V>::value &&                                                             \
                            detail::has_member_function_##_MEMBER_FUN_NAME_<V, Args...>::value),                      \
                           detail::has_member_function_##_MEMBER_FUN_NAME_##_t<V, Args...>>                           \
    {                                                                                                                 \
        return v._MEMBER_FUN_NAME_(std::forward<Args>(args)...);                                                      \
    };                                                                                                                \
    template <typename V, typename I>                                                                                 \
    auto _GLOBAL_FUN_(V&& v, std::initializer_list<I> const& list)                                                    \
        ->std::enable_if_t<(traits::is_array<V>::value &&                                                             \
                            detail::has_member_function_##_MEMBER_FUN_NAME_<V, std::initializer_list<I>>::value),     \
                           detail::has_member_function_##_MEMBER_FUN_NAME_##_t<V, std::initializer_list<I>>>          \
    {                                                                                                                 \
        return v._MEMBER_FUN_NAME_(list);                                                                             \
    };                                                                                                                \
    template <typename V, typename I>                                                                                 \
    auto _GLOBAL_FUN_(V& v, std::initializer_list<std::initializer_list<I>> const& list)                              \
        ->std::enable_if_t<                                                                                           \
            (traits::is_array<V>::value && detail::has_member_function_##_MEMBER_FUN_NAME_<                           \
                                               V, std::initializer_list<std::initializer_list<I>>>::value),           \
            detail::has_member_function_##_MEMBER_FUN_NAME_##_t<V, std::initializer_list<std::initializer_list<I>>>>  \
    {                                                                                                                 \
        return v._MEMBER_FUN_NAME_(list);                                                                             \
    };                                                                                                                \
    template <typename V, unsigned int... N, typename... Args>                                                        \
    auto _GLOBAL_FUN_(nTuple<V, N...>& v, Args&&... args)                                                             \
        ->std::enable_if_t<                                                                                           \
            (traits::is_array<V>::value && detail::has_member_function_##_MEMBER_FUN_NAME_<V, Args...>::value),       \
            nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<V, Args...>>, N...>>    \
    {                                                                                                                 \
        nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<V, Args...>>, N...> res;    \
        auto ib = begin(v), ie = end(v);                                                                              \
        auto jb = begin(res);                                                                                         \
        for (; ib != ie; ++ib, ++jb)                                                                                  \
        {                                                                                                             \
            *jb = _GLOBAL_FUN_(*ib, std::forward<Args>(args)...);                                                     \
        }                                                                                                             \
        return res;                                                                                                   \
    };                                                                                                                \
    template <typename V, unsigned int... N, typename I>                                                              \
    auto _GLOBAL_FUN_(nTuple<V, N...>& v, std::initializer_list<I> const& list)                                       \
        ->std::enable_if_t<(traits::is_array<V>::value &&                                                             \
                            detail::has_member_function_##_MEMBER_FUN_NAME_<V, std::initializer_list<I>>::value),     \
                           nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<         \
                                      V, std::initializer_list<I>>>,                                                  \
                                  N...>>                                                                              \
    {                                                                                                                 \
        nTuple<                                                                                                       \
            traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<V, std::initializer_list<I>>>, \
            N...>                                                                                                     \
            res;                                                                                                      \
        auto ib = begin(v), ie = end(v);                                                                              \
        auto jb = begin(res);                                                                                         \
        for (; ib != ie; ++ib, ++jb)                                                                                  \
        {                                                                                                             \
            *jb = _GLOBAL_FUN_(*ib, list);                                                                            \
        }                                                                                                             \
        return res;                                                                                                   \
    };                                                                                                                \
    template <typename V, unsigned int... N, typename I>                                                              \
    auto _GLOBAL_FUN_(nTuple<V, N...>& v, std::initializer_list<std::initializer_list<I>> const& list)                \
        ->std::enable_if_t<(traits::is_array<V>::value &&                                                             \
                            detail::has_member_function_##_MEMBER_FUN_NAME_<                                          \
                                V, std::initializer_list<std::initializer_list<I>>>::value),                          \
                           nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<         \
                                      V, std::initializer_list<std::initializer_list<I>>>>,                           \
                                  N...>>                                                                              \
    {                                                                                                                 \
        nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<                            \
                   V, std::initializer_list<std::initializer_list<I>>>>,                                              \
               N...>                                                                                                  \
            res;                                                                                                      \
        auto ib = begin(v), ie = end(v);                                                                              \
        auto jb = begin(res);                                                                                         \
        for (; ib != ie; ++ib, ++jb)                                                                                  \
        {                                                                                                             \
            *jb = _GLOBAL_FUN_(*ib, list);                                                                            \
        }                                                                                                             \
        return res;                                                                                                   \
    };                                                                                                                \
    template <typename V, unsigned int... N, typename... Args>                                                        \
    auto _GLOBAL_FUN_(nTuple<V, N...> const& v, Args&&... args)                                                       \
        ->std::enable_if_t<                                                                                           \
            (traits::is_array<V>::value && detail::has_member_function_##_MEMBER_FUN_NAME_<const V, Args...>::value), \
            nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<const V, Args...>>,     \
                   N...>>                                                                                             \
    {                                                                                                                 \
        nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<const V, Args...>>, N...>   \
            res;                                                                                                      \
        auto ib = begin(v), ie = end(v);                                                                              \
        auto jb = begin(res);                                                                                         \
        for (; ib != ie; ++ib, ++jb)                                                                                  \
        {                                                                                                             \
            *jb = _GLOBAL_FUN_(*ib, std::forward<Args>(args)...);                                                     \
        }                                                                                                             \
        return res;                                                                                                   \
    };                                                                                                                \
    template <typename V, unsigned int... N, typename I>                                                              \
    auto _GLOBAL_FUN_(nTuple<V, N...> const& v, std::initializer_list<I> const& list)                                 \
        ->std::enable_if_t<                                                                                           \
            (traits::is_array<V>::value &&                                                                            \
             detail::has_member_function_##_MEMBER_FUN_NAME_<const V, std::initializer_list<I>>::value),              \
            nTuple<traits::remove_cvref_t<                                                                            \
                       detail::has_member_function_##_MEMBER_FUN_NAME_##_t<const V, std::initializer_list<I>>>,       \
                   N...>>                                                                                             \
    {                                                                                                                 \
        nTuple<traits::remove_cvref_t<                                                                                \
                   detail::has_member_function_##_MEMBER_FUN_NAME_##_t<const V, std::initializer_list<I>>>,           \
               N...>                                                                                                  \
            res;                                                                                                      \
        auto ib = begin(v), ie = end(v);                                                                              \
        auto jb = begin(res);                                                                                         \
        for (; ib != ie; ++ib, ++jb)                                                                                  \
        {                                                                                                             \
            *jb = _GLOBAL_FUN_(*ib, list);                                                                            \
        }                                                                                                             \
        return res;                                                                                                   \
    };                                                                                                                \
    template <typename V, unsigned int... N, typename I>                                                              \
    auto _GLOBAL_FUN_(nTuple<V, N...> const& v, std::initializer_list<std::initializer_list<I>> const& list)          \
        ->std::enable_if_t<(traits::is_array<V>::value &&                                                             \
                            detail::has_member_function_##_MEMBER_FUN_NAME_<                                          \
                                const V, std::initializer_list<std::initializer_list<I>>>::value),                    \
                           nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<         \
                                      const V, std::initializer_list<std::initializer_list<I>>>>,                     \
                                  N...>>                                                                              \
    {                                                                                                                 \
        nTuple<traits::remove_cvref_t<detail::has_member_function_##_MEMBER_FUN_NAME_##_t<                            \
                   const V, std::initializer_list<std::initializer_list<I>>>>,                                        \
               N...>                                                                                                  \
            res;                                                                                                      \
        auto ib = begin(v), ie = end(v);                                                                              \
        auto jb = begin(res);                                                                                         \
        for (; ib != ie; ++ib, ++jb)                                                                                  \
        {                                                                                                             \
            *jb = _GLOBAL_FUN_(*ib, list);                                                                            \
        }                                                                                                             \
        return res;                                                                                                   \
    };

DEFINE_NTUPLE_FOREACH_MEMBER_METHOD(reshape, reshape)
DEFINE_NTUPLE_FOREACH_MEMBER_METHOD(initialize, initialize)
DEFINE_NTUPLE_FOREACH_MEMBER_METHOD(fill, fill)
DEFINE_NTUPLE_FOREACH_MEMBER_FUNCTION(get_by_idx, get)
DEFINE_NTUPLE_FOREACH_MEMBER_FUNCTION(slice, slice)
DEFINE_NTUPLE_FOREACH_MEMBER_FUNCTION(shift, shift)

template <typename... V>
auto begin(Array<V...>& a)
{
    return a.begin();
}
template <typename... V>
auto end(Array<V...>& a)
{
    return a.end();
}
template <typename... V>
auto begin(Array<V...> const& a)
{
    return a.begin();
}
template <typename... V>
auto end(Array<V...> const& a)
{
    return a.end();
}

namespace utility
{
template <typename V, typename SFC>
void swap(Array<V, SFC>& lhs, Array<V, SFC>& rhs) { lhs.swap(rhs); }

template <typename... T, typename V>
V fill(Array<T...>& a, V start, V inc = 0)
{
    for (decltype(auto) item : a)
    {
        start = fill(item, start, inc);
    }
    return start;
}
template <typename... T, typename V>
V fill(Array<T...>&& a, V start, V inc = 0)
{
    for (decltype(auto) item : a)
    {
        start = fill(item, start, inc);
    }
    return start;
}
} // namespace utility

template <typename V, typename SFC>
std::ostream& operator<<(std::ostream& os, Array<V, SFC> const& v)
{
    utility::FancyPrintP(os, v.begin(), v.ndim, &v.count()[0], 0, 4);
    return os;
};

template <typename V, typename SFC>
std::istream& operator>>(std::istream& is, Array<V, SFC>& lhs)
{
    //    UNIMPLEMENTED;
    return is;
};

#define _SP_DEFINE_BINARY_FUNCTION(_TAG_, _FUN_)                                                                       \
    template <typename TL, typename TR>                                                                                \
    auto _FUN_(TL const& lhs, TR const& rhs)                                                                           \
        ->std::enable_if_t<                                                                                            \
            (traits::is_array<TL>::value && traits::is_array<TR>::value) ||                                            \
                (traits::is_array<TL>::value && !(traits::is_array<TR>::value || traits::is_expression<TR>::value)) || \
                (!(traits::is_array<TL>::value || traits::is_expression<TL>::value) && traits::is_array<TR>::value),   \
            Expression<tags::_TAG_, TL, TR>>                                                                           \
    {                                                                                                                  \
        return Expression<tags::_TAG_, TL, TR>(lhs, rhs);                                                              \
    };

#define _SP_DEFINE_UNARY_FUNCTION(_TAG_, _OP_)                                                              \
    template <typename TL>                                                                                  \
    auto _FUN_(TL const& lhs)->std::enable_if_t<(traits::is_array<TL>::value), Expression<tags::_TAG_, TL>> \
    {                                                                                                       \
        return Expression<tags::_TAG_, TL>(lhs);                                                            \
    };

_SP_DEFINE_BINARY_FUNCTION(addition, operator+)
_SP_DEFINE_BINARY_FUNCTION(subtraction, operator-)
_SP_DEFINE_BINARY_FUNCTION(multiplication, operator*)
_SP_DEFINE_BINARY_FUNCTION(division, operator/)
_SP_DEFINE_BINARY_FUNCTION(modulo, operator%)

_SP_DEFINE_UNARY_FUNCTION(bitwise_not, operator~)
_SP_DEFINE_BINARY_FUNCTION(bitwise_xor, operator^)
_SP_DEFINE_BINARY_FUNCTION(bitwise_and, operator&)
_SP_DEFINE_BINARY_FUNCTION(bitwise_or, operator|)

//_SP_DEFINE_BINARY_FUNCTION(bitwise_left_shift, <<)
//_SP_DEFINE_BINARY_FUNCTION(bitwise_right_shifit, >>)

_SP_DEFINE_UNARY_FUNCTION(unary_plus, operator+)
_SP_DEFINE_UNARY_FUNCTION(unary_minus, operator-)

_SP_DEFINE_UNARY_FUNCTION(logical_not, operator!)
_SP_DEFINE_BINARY_FUNCTION(logical_and, operator&&)
_SP_DEFINE_BINARY_FUNCTION(logical_or, operator||)

_SP_DEFINE_UNARY_FUNCTION(cos, cos)
_SP_DEFINE_UNARY_FUNCTION(acos, acos)
_SP_DEFINE_UNARY_FUNCTION(cosh, cosh)
_SP_DEFINE_UNARY_FUNCTION(sin, sin)
_SP_DEFINE_UNARY_FUNCTION(asin, asin)
_SP_DEFINE_UNARY_FUNCTION(sinh, sinh)
_SP_DEFINE_UNARY_FUNCTION(tan, tan)
_SP_DEFINE_UNARY_FUNCTION(tanh, tanh)
_SP_DEFINE_UNARY_FUNCTION(atan, atan)
_SP_DEFINE_UNARY_FUNCTION(exp, exp)
_SP_DEFINE_UNARY_FUNCTION(log, log)
_SP_DEFINE_UNARY_FUNCTION(log10, log10)
_SP_DEFINE_UNARY_FUNCTION(sqrt, sqrt)
_SP_DEFINE_BINARY_FUNCTION(atan2, atan2)
_SP_DEFINE_BINARY_FUNCTION(pow, pow)

#undef _SP_DEFINE_BINARY_FUNCTION
#undef _SP_DEFINE_UNARY_FUNCTION

template <typename... TL>
auto operator<<(Array<TL...> const& lhs, unsigned int n)
{
    return Expression<tags::bitwise_left_shift, Array<TL...>, int>(lhs, n);
};

template <typename... TL>
auto operator>>(Array<TL...> const& lhs, unsigned int n)
{
    return Expression<tags::bitwise_right_shifit, Array<TL...>, int>(lhs, n);
};

#define _SP_DEFINE_COMPOUND_OP(_TAG_, _FUN_)                                                 \
    template <typename TL, typename TR>                                                      \
    auto _FUN_(TL& lhs, TR const& rhs)->std::enable_if_t<(traits::is_array<TL>::value), TL&> \
    {                                                                                        \
        calculus::evaluate_expression(lhs, Expression<tags::_TAG_, TL, TR>(lhs, rhs));       \
        return lhs;                                                                          \
    }

_SP_DEFINE_COMPOUND_OP(addition, operator+=)
_SP_DEFINE_COMPOUND_OP(subtraction, operator-=)
_SP_DEFINE_COMPOUND_OP(multiplication, operator*=)
_SP_DEFINE_COMPOUND_OP(division, operator/=)
_SP_DEFINE_COMPOUND_OP(modulo, operator%=)
_SP_DEFINE_COMPOUND_OP(bitwise_xor, operator^=)
_SP_DEFINE_COMPOUND_OP(bitwise_and, operator&=)
_SP_DEFINE_COMPOUND_OP(bitwise_or, operator|=)
_SP_DEFINE_COMPOUND_OP(bitwise_left_shift, operator<<=)
_SP_DEFINE_COMPOUND_OP(bitwise_left_shift, operator>>=)

#undef _SP_DEFINE_COMPOUND_OP

#define _SP_DEFINE_BINARY_BOOLEAN_FUNCTION(_TAG_, _REDUCTION_, _FUN_)                                              \
    template <typename TL, typename TR>                                                                            \
    std::enable_if_t<                                                                                              \
        (traits::is_array<TL>::value && traits::is_array<TR>::value) ||                                            \
            (traits::is_array<TL>::value && !(traits::is_array<TR>::value || traits::is_expression<TR>::value)) || \
            (!(traits::is_array<TL>::value || traits::is_expression<TL>::value) && traits::is_array<TR>::value),   \
        bool>                                                                                                      \
    _FUN_(TL const& lhs, TR const& rhs)                                                                            \
    {                                                                                                              \
        return calculus::reduce_expression<tags::_REDUCTION_>(Expression<tags::_TAG_, TL, TR>(lhs, rhs));          \
    };

_SP_DEFINE_BINARY_BOOLEAN_FUNCTION(not_equal_to, logical_or, operator!=)
_SP_DEFINE_BINARY_BOOLEAN_FUNCTION(equal_to, logical_and, operator==)
_SP_DEFINE_BINARY_BOOLEAN_FUNCTION(less_equal, logical_and, operator<=)
_SP_DEFINE_BINARY_BOOLEAN_FUNCTION(greater_equal, logical_and, operator>=)
_SP_DEFINE_BINARY_BOOLEAN_FUNCTION(greater, logical_and, operator>)
_SP_DEFINE_BINARY_BOOLEAN_FUNCTION(less, logical_and, operator<)

#undef _SP_DEFINE_BINARY_BOOLEAN_FUNCTION

} // namespace sp

#endif // SP_ARRAY_H
