//
// Created by salmon on 18-2-27.
//

#ifndef SP_NTUPLE_H
#define SP_NTUPLE_H

#include "ExpressionTemplate.h"
#include "TypeTraits.h"
#include <cassert>
#include <cstring> //for memcpy
#include <iomanip>
#include <iostream>
#include <limits>

namespace sp
{

/**
 * @brief nTupleBasic
 *  nTuple is a container that encapsulates fixed size nd-arrays
 *  nTuple<V,N>    => std::array<V,N>
 *  nTuple<V,N,M>  => std::array<std::array<V,M>,N>
 * @ref   An n-tuple is a sequence (or ordered list) of n elements, where n is a non-negative integer. There is only one
 * 0-tuple, an empty sequence. An n-tuple is defined inductively using the construction of an ordered pair.
 * @tparam TV
 * @tparam IS_OWNED
 * @tparam ...
 */
template <typename TV, unsigned int...>
struct nTupleBasic;
template <typename TV, unsigned int...>
struct nTupleIterator;
template <typename...>
class Expression;

namespace traits
{
// std::
template <typename V, unsigned int OWEN, unsigned int... N>
struct rank<nTupleBasic<V, OWEN, N...>> : public std::integral_constant<std::size_t, rank<V>::value + sizeof...(N)>
{
};
template <typename V, unsigned int OWEN, unsigned int N0, unsigned int... N>
struct extent<nTupleBasic<V, OWEN, N0, N...>, 0> : public std::integral_constant<std::size_t, N0>
{
};
template <typename V, unsigned int OWEN>
struct extent<nTupleBasic<V, OWEN>, 0> : public std::integral_constant<std::size_t, 1>
{
};
template <unsigned int _Uint, typename V, unsigned int OWEN, unsigned int N0, unsigned int... N>
struct extent<nTupleBasic<V, OWEN, N0, N...>, _Uint>
    : public std::integral_constant<std::size_t, extent<nTupleBasic<V, OWEN, N...>, _Uint - 1>::value>
{
};
template <typename V, unsigned int OWEN, unsigned int N0, unsigned int... N>
struct remove_extent<nTupleBasic<V, OWEN, N0, N...>>
{
    typedef std::conditional_t<sizeof...(N) == 0, V, nTupleBasic<V, OWEN, N...>> type;
};
template <typename V, unsigned int OWEN, unsigned int... N>
struct remove_all_extents<nTupleBasic<V, OWEN, N...>>
{
    typedef remove_all_extents_t<V> type;
};

// not std
template <typename V>
struct is_nTuple : public std::integral_constant<bool, false>
{
};
template <typename V>
struct is_nTuple<V&> : public std::integral_constant<bool, is_nTuple<V>::value>
{
};
template <typename V>
struct is_nTuple<V const> : public std::integral_constant<bool, is_nTuple<V>::value>
{
};
template <typename V>
struct is_nTuple<V const&> : public std::integral_constant<bool, is_nTuple<V>::value>
{
};
template <typename V>
struct is_nTuple<V&&> : public std::integral_constant<bool, is_nTuple<V>::value>
{
};
template <typename TV, unsigned int... N>
struct is_nTuple<nTupleBasic<TV, N...>> : public std::integral_constant<bool, true>
{
};

template <typename TV, unsigned int... N>
struct add_reference<nTupleBasic<TV, true, N...>>
{
    typedef nTupleBasic<TV, true, N...> const& type;
};
template <typename TV, unsigned int... N>
struct add_reference<nTupleBasic<TV, false, N...>>
{
    typedef nTupleBasic<TV, false, N...> type;
};

// template <typename V, unsigned int N>
// nTupleBasic<std::size_t, true, N> nested_initializer_list_traits_dims(make_nested_initializer_list<V, N> const& list)
// {
//    nTupleBasic<std::size_t, true, N> res;
//    res = 0;
//    nested_initializer_list_traits_dims<V, N>(res.m_data_, list);
//    return res;
//};

} // namespace traits

namespace calculus
{
template <typename V, unsigned int... _UInt>
struct RecursiveGetHelper<nTupleBasic<V, _UInt...>>
{
    template <std::size_t I, typename U>
    static decltype(auto) eval(U&& v)
    {
        return v.m_data_[I];
    }
};
template <typename V>
struct RecursiveGetHelper<V, std::enable_if_t<(traits::rank<V>::value == 1 && !traits::is_expression<V>::value &&
                                               !traits::is_nTuple<V>::value)>>
{
    template <std::size_t I, typename U>
    static decltype(auto) eval(U&& v)
    {
        return v[I];
    }
};
template <typename V>
struct RecursiveGetHelper<V, std::enable_if_t<(traits::rank<V>::value > 1 && !traits::is_expression<V>::value &&
                                               !traits::is_nTuple<V>::value)>>
{
    template <std::size_t I, typename U>
    static decltype(auto) eval(U&& v)
    {
        static constexpr std::size_t N = traits::number_of_elements<V, 1>::value;
        static_assert(N > 1, "N<=1");
        return get_r<static_cast<size_t>(I % N)>(v[(static_cast<std::size_t>(I / N))]);
    }
};

} // namespace calculus

/**
 *  0-tuple
 * @tparam TV
 * @tparam IS_OWNED
 */
template <typename TV, unsigned int IS_OWNED>
struct nTupleBasic<TV, IS_OWNED>
{
    typedef nTupleBasic<TV, IS_OWNED> this_type;

    typedef TV value_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef value_type* iterator;
    typedef value_type const* const_iterator;
    typedef std::conditional_t<IS_OWNED == 1, value_type[1], value_type*> data_type;

    data_type m_data_;

    value_type* data() noexcept { return &m_data_[0]; }
    value_type const* data() const noexcept { return &m_data_[0]; }
    constexpr std::size_t size() const noexcept { return 1; }
    constexpr std::size_t max_size() const noexcept { return 1; }
    constexpr bool empty() const noexcept { return false; }

    constexpr reference get(std::size_t s) { return m_data_[0]; };
    constexpr const_reference get(std::size_t s) const { return m_data_[0]; }

    constexpr reference operator[](std::size_t s) { return get(s); };
    constexpr const_reference operator[](std::size_t s) const { return get(s); };
    constexpr reference at(std::size_t s)
    {
        if (s > 0)
        {
            throw(std::out_of_range("nTuple"));
        }
        return get(s);
    };
    constexpr const_reference at(std::size_t s) const
    {
        if (s > 0)
        {
            throw(std::out_of_range("nTuple"));
        }
        return get(s);
    };

    template <typename U>
    this_type& operator=(U const& other)
    {
        calculus::evaluate_expression(*this, other);
        return *this;
    }

    template <typename U>
    this_type& operator=(std::initializer_list<U> const& list)
    {
        calculus::evaluate_expression(*this, list);
        return *this;
    }
};
/**
 *  nested  n-tuple: ((a,b,c...),(a,b,c...),(a,b,c...) ...)
 */
template <typename TV, unsigned int IS_OWNED, unsigned int N0, unsigned int... N>
struct nTupleBasic<TV, IS_OWNED, N0, N...>
{
private:
    typedef nTupleBasic<TV, IS_OWNED, N0, N...> this_type;

public:
    static constexpr unsigned int rank = sizeof...(N) + 1;
    typedef TV value_type;
    typedef std::conditional_t<rank == 1, value_type&, nTupleBasic<value_type, false, N...>> reference;
    typedef std::conditional_t<rank == 1, value_type const&, nTupleBasic<std::add_const_t<value_type>, false, N...>>
        const_reference;
    typedef std::conditional_t<rank == 1, value_type*, nTupleIterator<value_type, N0, N...>> iterator;
    typedef std::conditional_t<rank == 1, value_type const*, nTupleIterator<const value_type, N0, N...>> const_iterator;
    typedef std::conditional_t<IS_OWNED == 1, value_type[traits::number_of_elements<this_type>::value], value_type*>
        data_type;

    data_type m_data_;

    nTupleBasic() {}
    template <typename U, unsigned int... M>
    nTupleBasic(nTupleBasic<U, M...> const& other) : nTupleBasic(other.m_data_) {}
    nTupleBasic(this_type const& other) : nTupleBasic(other.m_data_) {}
    nTupleBasic(this_type&& other) : nTupleBasic(other.m_data_) {}

    template <typename U,
              typename std::enable_if_t<(std::is_same<U, value_type>::value) && (IS_OWNED == false)>* = nullptr>
    nTupleBasic(U* d, size_t len = traits::number_of_elements<this_type>::value) : m_data_(d)
    {
        assert(len >= traits::number_of_elements<this_type>::value);
    }
    template <typename U,
              typename std::enable_if_t<!(std::is_same<U, value_type>::value) || (IS_OWNED == true)>* = nullptr>
    nTupleBasic(U* d, size_t len = traits::number_of_elements<this_type>::value)
    {
        memcpy(reinterpret_cast<void*>(m_data_), reinterpret_cast<void*>(traits::remove_const(d)),
               sizeof(value_type) * std::min(len, traits::number_of_elements<this_type>::value));
    }
    template <typename U>
    nTupleBasic(traits::make_nested_initializer_list<U, rank> const& list)
    {
        calculus::evaluate_expression(*this, list);
    }

    nTupleBasic(traits::make_nested_initializer_list<value_type, rank> const& list)
    {
        calculus::evaluate_expression(*this, list);
    }

    operator nTupleBasic<value_type, true, N...>() const { return nTupleBasic<value_type, true, N...>(*this); };

    constexpr value_type* data() noexcept { return &m_data_[0]; }
    constexpr value_type const* data() const noexcept { return &m_data_[0]; }
    constexpr std::size_t size() const noexcept { return traits::number_of_elements<this_type>::value; }
    constexpr std::size_t max_size() const noexcept { return traits::number_of_elements<this_type>::value; }
    constexpr bool empty() const noexcept { return max_size() == 0; }

private:
    constexpr decltype(auto) get_(std::size_t s, std::false_type) { return m_data_[s]; };
    constexpr decltype(auto) get_(std::size_t s, std::false_type) const { return m_data_[s]; };
    constexpr decltype(auto) get_(std::size_t s, std::true_type)
    {
        return reference{&m_data_[s * traits::number_of_elements<reference>::value]};
    };
    constexpr decltype(auto) get_(std::size_t s, std::true_type) const
    {
        return const_reference{&m_data_[s * traits::number_of_elements<const_reference>::value]};
    };

public:
    constexpr reference get(std::size_t s) { return get_(s, std::integral_constant<bool, (rank > 1)>()); };
    constexpr const_reference get(std::size_t s) const { return get_(s, std::integral_constant<bool, (rank > 1)>()); }

    constexpr reference operator[](std::size_t s) { return get(s); };
    constexpr const_reference operator[](std::size_t s) const { return get(s); };
    constexpr reference at(std::size_t s)
    {
        if (s >= N0)
        {
            throw(std::out_of_range("nTuple"));
        }
        return get(s);
    };
    constexpr const_reference at(std::size_t s) const
    {
        if (s >= N0)
        {
            throw(std::out_of_range("nTuple"));
        }
        return get(s);
    };

    template <typename U>
    this_type& operator=(U const& other)
    {
        calculus::evaluate_expression(*this, other);
        return *this;
    }
    template <typename U>
    this_type& operator=(traits::make_nested_initializer_list<U, rank> const& list)
    {
        calculus::evaluate_expression(*this, list);
        return *this;
    }
    this_type& operator=(this_type const& other)
    {
        calculus::evaluate_expression(*this, other);
        return *this;
    }
};

template <typename TV, unsigned int N0, unsigned int... N>
struct nTupleIterator<TV, N0, N...>
{
    typedef std::random_access_iterator_tag iterator_category;
    typedef nTupleBasic<TV, false, N...> value_type;
    typedef std::ptrdiff_t difference_type;
    typedef nTupleIterator<TV, N...> pointer;
    typedef nTupleBasic<TV, false, N...> reference;

private:
    TV* m_data_ = nullptr;
    typedef nTupleIterator<TV, N0, N...> this_type;

public:
    explicit nTupleIterator(TV* d) : m_data_(d) {}
    nTupleIterator() = default;
    ~nTupleIterator() = default;

    this_type& operator++()
    {
        Next();
        return *this;
    }
    this_type operator++(int)
    {
        this_type tmp(*this);
        Next();
        return tmp;
    }
    this_type& operator--()
    {
        Prev();
        return *this;
    }
    this_type operator--(int)
    {
        this_type tmp(*this);
        Prev();
        return tmp;
    }
    this_type operator+(std::size_t n)
    {
        this_type tmp(*this);
        Next(n);
        return tmp;
    }
    this_type operator-(std::size_t n)
    {
        this_type tmp(*this);
        Prev(n);
        return tmp;
    }
    this_type& operator+=(std::size_t n)
    {
        Next(n);
        return *this;
    }
    this_type& operator-=(std::size_t n)
    {
        Prev(n);
        return *this;
    }
    bool operator==(this_type const& other) { return m_data_ == other.m_data_; }
    bool operator!=(this_type const& other) { return m_data_ != other.m_data_; }

    reference operator*() { return reference{m_data_}; }
    pointer operator->() { return pointer{m_data_}; }

private:
    void Next(std::size_t n = 1) { m_data_ += n * traits::number_of_elements<reference>::value; }
    void Prev(std::size_t n = 1) { m_data_ -= n * traits::number_of_elements<reference>::value; }
};

template <typename TV, unsigned int IS_OWNED>
auto begin(nTupleBasic<TV, IS_OWNED>& t)
{
    return &t.m_data_[0];
};
template <typename TV, unsigned int IS_OWNED>
auto end(nTupleBasic<TV, IS_OWNED>& t)
{
    return &t.m_data_[1];
};
template <typename TV, unsigned int IS_OWNED>
auto begin(nTupleBasic<TV, IS_OWNED> const& t)
{
    return &t.m_data_[0];
};
template <typename TV, unsigned int IS_OWNED>
auto end(nTupleBasic<TV, IS_OWNED> const& t)
{
    return &t.m_data_[1];
};

template <typename TV, unsigned int IS_OWNED, unsigned int N0>
auto begin(nTupleBasic<TV, IS_OWNED, N0>& t)
{
    return t.m_data_;
};
template <typename TV, unsigned int IS_OWNED, unsigned int N0>
auto end(nTupleBasic<TV, IS_OWNED, N0>& t)
{
    return t.m_data_ + N0;
};
template <typename TV, unsigned int IS_OWNED, unsigned int N0>
auto begin(nTupleBasic<TV, IS_OWNED, N0> const& t)
{
    return t.m_data_;
};
template <typename TV, unsigned int IS_OWNED, unsigned int N0>
auto end(nTupleBasic<TV, IS_OWNED, N0> const& t)
{
    return t.m_data_ + N0;
};

template <typename TV, unsigned int IS_OWNED, unsigned int... N>
auto begin(nTupleBasic<TV, IS_OWNED, N...>& t)
{
    return nTupleIterator<TV, N...>(t.m_data_);
};
template <typename TV, unsigned int IS_OWNED, unsigned int... N>
auto end(nTupleBasic<TV, IS_OWNED, N...>& t)
{
    return nTupleIterator<TV, N...>(t.m_data_ + traits::number_of_elements<nTupleBasic<TV, IS_OWNED, N...>>::value);
};
template <typename TV, unsigned int IS_OWNED, unsigned int... N>
auto begin(nTupleBasic<TV, IS_OWNED, N...> const& t)
{
    return nTupleIterator<const TV, N...>(t.m_data_);
};
template <typename TV, unsigned int IS_OWNED, unsigned int... N>
auto end(nTupleBasic<TV, IS_OWNED, N...> const& t)
{
    return nTupleIterator<const TV, N...>(t.m_data_ +
                                          traits::number_of_elements<nTupleBasic<TV, IS_OWNED, N...>>::value);
};

template <typename TV, unsigned int IS_OWNED, unsigned int... N>
std::ostream& operator<<(std::ostream& os, nTupleBasic<TV, IS_OWNED, N...> const& t)
{
    return utility::FancyPrint(os, t, 0);
};
template <typename V, unsigned int... N>
std::istream& operator>>(std::istream& is, nTupleBasic<V, N...>& d)
{
    //    for (int i = 0, ie = traits::number_of_elements<nTupleBasic<V, N...>>::value; i < ie; ++i) { is >>
    //    d.m_data_[i]; }
    for (auto& v : d)
    {
        is >> v;
    }
    return is;
}

template <typename TV, unsigned int... N>
using nTuple = nTupleBasic<TV, true, N...>;
template <typename TV, unsigned int... N>
using nTupleView = nTupleBasic<TV, false, N...>;
#define _SP_DEFINE_BINARY_FUNCTION(_TAG_, _FUN_)                                                  \
    template <typename TL, typename TR>                                                           \
    auto _FUN_(TL const& lhs, TR const& rhs)                                                      \
        ->std::enable_if_t<(traits::is_nTuple<TL>::value && traits::is_nTuple<TR>::value) ||      \
                               (traits::is_nTuple<TL>::value && std::is_arithmetic<TR>::value) || \
                               (std::is_arithmetic<TL>::value && traits::is_nTuple<TR>::value),   \
                           Expression<tags::_TAG_, TL, TR>>                                       \
    {                                                                                             \
        return Expression<tags::_TAG_, TL, TR>(lhs, rhs);                                         \
    };

#define _SP_DEFINE_UNARY_FUNCTION(_TAG_, _FUN_)                                               \
    template <typename TL>                                                                    \
    __host__ __device__ auto _FUN_(TL const& lhs)                                             \
        ->std::enable_if_t<(traits::is_nTuple<TL>::value), Expression<tags::_TAG_, const TL>> \
    {                                                                                         \
        return Expression<sp::tags::_TAG_, const TL>(lhs);                                    \
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

template <typename TL, unsigned int... NL>
__host__ __device__ auto operator<<(nTupleBasic<TL, NL...> const& lhs, unsigned int rhs)
{
    return Expression<sp::tags::bitwise_left_shift, nTupleBasic<TL, NL...>, unsigned int>(lhs, rhs);
};
template <typename TL, unsigned int... NL>
__host__ __device__ auto operator>>(nTupleBasic<TL, NL...> const& lhs, unsigned int rhs)
{
    return Expression<sp::tags::bitwise_right_shifit, nTupleBasic<TL, NL...>, unsigned int>(lhs, rhs);
};

#define _SP_DEFINE_COMPOUND_FUN(_TAG_, _FUN_)                                                                         \
    template <typename TL, typename TR>                                                                               \
    __host__ __device__ auto _FUN_(TL&& lhs, TR const& rhs)->std::enable_if_t<traits::is_nTuple<TL>::value, TL&>      \
    {                                                                                                                 \
        calculus::evaluate_expression(std::forward<TL>(lhs), Expression<tags::_TAG_, traits::remove_cvref_t<TL>, TR>( \
                                                                 std::forward<TL>(lhs), rhs));                        \
        return lhs;                                                                                                   \
    }

_SP_DEFINE_COMPOUND_FUN(addition, operator+=)
_SP_DEFINE_COMPOUND_FUN(subtraction, operator-=)
_SP_DEFINE_COMPOUND_FUN(multiplication, operator*=)
_SP_DEFINE_COMPOUND_FUN(division, operator/=)
_SP_DEFINE_COMPOUND_FUN(modulo, operator%=)

_SP_DEFINE_COMPOUND_FUN(bitwise_xor, operator^=)
_SP_DEFINE_COMPOUND_FUN(bitwise_and, operator&=)
_SP_DEFINE_COMPOUND_FUN(bitwise_or, operator|=)

_SP_DEFINE_COMPOUND_FUN(bitwise_left_shift, operator<<=)
_SP_DEFINE_COMPOUND_FUN(bitwise_right_shifit, operator>>=)

#undef _SP_DEFINE_COMPOUND_FUN

#define _SP_DEFINE_BINARY_BOOLEAN_OPERATOR(_TOP_, _REDUCTION_, _OP_)                                      \
    template <typename TL, typename TR>                                                                   \
    __host__ __device__ auto _OP_(TL const& lhs, TR const& rhs)                                           \
        ->std::enable_if_t<(traits::is_nTuple<TL>::value || traits::is_nTuple<TR>::value), bool>          \
    {                                                                                                     \
        return calculus::reduce_expression<tags::_REDUCTION_>(Expression<tags::_TOP_, TL, TR>(lhs, rhs)); \
    };
_SP_DEFINE_BINARY_BOOLEAN_OPERATOR(not_equal_to, logical_or, operator!=)
_SP_DEFINE_BINARY_BOOLEAN_OPERATOR(equal_to, logical_and, operator==)
_SP_DEFINE_BINARY_BOOLEAN_OPERATOR(less, logical_and, operator<=)
_SP_DEFINE_BINARY_BOOLEAN_OPERATOR(greater, logical_and, operator>=)
_SP_DEFINE_BINARY_BOOLEAN_OPERATOR(less_equal, logical_and, operator>)
_SP_DEFINE_BINARY_BOOLEAN_OPERATOR(greater_equal, logical_and, operator<)
#undef _SP_DEFINE_BINARY_BOOLEAN_OPERATOR

namespace utility
{
template <typename TV, unsigned int O0, unsigned int O1, unsigned int... N>
nTupleBasic<TV, true, 2, N...> tie(nTupleBasic<TV, O0, N...> const& first, nTupleBasic<TV, O1, N...> const& second)
{
    nTupleBasic<TV, true, 2, N...> res;
    res[0] = first;
    res[1] = second;
    return std::move(res);
};
template <typename TV, typename... Others>
auto tie(TV const& first, Others&&... others)
    -> std::enable_if_t<std::is_arithmetic<TV>::value, nTuple<TV, 1 + sizeof...(Others)>>
{
    return nTuple<TV, 1 + sizeof...(Others)>{first, std::forward<Others>(others)...};
};
} // namespace utility

} // namespace sp
#endif // SP_NTUPLE_H
